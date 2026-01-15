#pragma once
#include "cudaKernel/myStruct/particle.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "cudaKernel/myStruct/interaction.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"
#include "cudaKernel/ballIntegration.h"

class ballHandler
{
public:
    ballHandler()
    {
        uploadFlag_ = false;
    }

    ~ballHandler() = default;

    void addCluster(std::vector<double3> positions, 
    std::vector<double3> velocities, 
    std::vector<double3> angularVelocities, 
    std::vector<double> radius, 
    double density, 
    int materialID,
    cudaStream_t stream,
    int clumpID = -1)
    {
        if(!uploadFlag_) 
        {
            uploadFlag_ = true;
            balls_.download(stream);
        }
        for (size_t i = 0; i < positions.size(); i++)
        {
            double mass = 4.0 / 3.0 * pi() * pow(radius[i], 3.0) * density;
            double invMass = 0.0;
            if(mass > 1.e-20) invMass = 1. / mass;
            balls_.addHost(positions[i], 
            velocities[i], 
            angularVelocities[i], 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            radius[i], 
            invMass, 
            materialID, 
            clumpID);
        }
    }

    void addFixedCluster(std::vector<double3> positions, 
    std::vector<double> radius, 
    int materialID,
    cudaStream_t stream,
    int clumpID = -1)
    {
        if(!uploadFlag_)
        {
            uploadFlag_ = true;
            balls_.download(stream);
        }
        for (size_t i = 0; i < positions.size(); i++)
        {
            balls_.addHost(positions[i], 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            radius[i], 
            0.0, 
            materialID, 
            clumpID);
        }
    }

    void addBondedObjects(const std::vector<int> &object0, const std::vector<int> &object1, cudaStream_t stream)
    {
        bondedBallInteractions_.add(object0,
        object1,
        balls_.positionVector(),
        stream);
    }

    void addExternalForce(const std::vector<double3>& externalForce, cudaStream_t stream)
    {
        if(externalForce.size() > balls_.hostSize()) return;
        std::vector<double3> force = balls_.forceVector();
        std::vector<double3> totalF(balls_.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(externalForce.begin(), 
        externalForce.end(),
        force.begin(), 
        totalF.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        balls_.setForceVector(totalF, stream);
    }

    void addExternalTorque(const std::vector<double3>& externalTorque, cudaStream_t stream)
    {
        if(externalTorque.size() > balls_.hostSize()) return;
        std::vector<double3> torque = balls_.torqueVector();
        std::vector<double3> totalT(balls_.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(externalTorque.begin(), 
        externalTorque.end(),
        torque.begin(), 
        totalT.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        balls_.setTorqueVector(totalT, stream);
    }

    ball& getBalls() {return balls_;}

    solidInteraction& getBallInteractions() {return ballInteractions_;}

    void upload(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if(uploadFlag_)
        {
            balls_.upload(stream);

            ballInteractions_.alloc(balls_.deviceSize() * 6, stream);
            ballInteractionMap_.alloc(balls_.deviceSize(), 
            balls_.deviceSize(), 
            balls_.deviceSize() * 6, 
            stream);
            double cellSizeOneDim = 0.0;
            std::vector<double> rad = balls_.radiusVector();
            if(rad.size() > 0) cellSizeOneDim = *std::max_element(rad.begin(), rad.end()) * 2.0 * 1.1;
            spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream);
            uploadFlag_ = false;
        }
    }

    void outputBallVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/solidParticle_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = balls_.hostSize();
        std::vector<double3> p = balls_.positionVector();
        std::vector<double3> v = balls_.velocityVector();
        std::vector<double3> a = balls_.angularVelocityVector();
        const std::vector<double> r = balls_.radiusVector();
        const std::vector<int> materialID = balls_.materialIDVector();
        const std::vector<int> clumpID = balls_.clumpIDVector();
        
        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"radius\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << r[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"materialID\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << materialID[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"clumpID\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << clumpID[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "velocity"       , v     },
            { "angularVelocity", a     }
        };
        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < v.size(); ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

    void neighborSearch(const size_t maxThreads, cudaStream_t stream)
    {
        launchBallNeighborSearch(ballInteractions_, 
        ballInteractionMap_, 
        balls_,
        spatialGrids_,
        maxThreads,
        stream);
    }

    void integration1st(const double3 g, const double dt, const size_t gridDim, const size_t blockDim, cudaStream_t stream)
    {
        launchBall1stHalfIntegration(balls_, 
        g, 
        dt, 
        gridDim,
        blockDim,
        stream);
    }

    void contactCalculation(contactModelParameters& contactModelParams, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchBallContactCalculation(ballInteractions_, 
        bondedBallInteractions_, 
        balls_, 
        contactModelParams, 
        ballInteractionMap_,
        dt, 
        maxThreads, 
        stream);
    }

    void integration2nd(const double3 g, const double dt, const size_t gridDim, const size_t blockDim, cudaStream_t stream)
    {
        launchBall2ndHalfIntegration(balls_, 
        g, 
        dt, 
        gridDim,
        blockDim,
        stream);
    }

private:
    bool uploadFlag_;
    ball balls_;
    spatialGrid spatialGrids_;

    solidInteraction ballInteractions_;
    bondedInteraction bondedBallInteractions_;
	interactionMap ballInteractionMap_;
};