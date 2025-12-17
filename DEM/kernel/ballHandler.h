#pragma once
#include "cudaKernel/myStruct/particle.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"

class ballHandler
{
public:
    ballHandler()
    {
        downLoadFlag_ = false;
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
        if(!downLoadFlag_) 
        {
            downLoadFlag_ = true;
            balls_.upload(stream);
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
        if(!downLoadFlag_) 
        {
            downLoadFlag_ = true;
            balls_.upload(stream);
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

    ball& balls() {return balls_;}

    spatialGrid& spatialGrids() {return spatialGrids_;}

    void download(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if(downLoadFlag_)
        {
            balls_.download(stream);
            downLoadFlag_ = false;

            double cellSizeOneDim = 0.0;
            std::vector<double> rad = balls_.radiusVector();
            if(rad.size() > 0) cellSizeOneDim = *std::max_element(rad.begin(), rad.end()) * 2.0 * 1.1;
            if(cellSizeOneDim > spatialGrids_.cellSize.x 
            || cellSizeOneDim > spatialGrids_.cellSize.y 
            || cellSizeOneDim > spatialGrids_.cellSize.z)
            {
                spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream);
            }
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
        const std::vector<double3> p = balls_.positionVector();
        const std::vector<double3> v = balls_.velocityVector();
        const std::vector<double3> a = balls_.angularVelocityVector();
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

private:
    bool downLoadFlag_;
    ball balls_;
    spatialGrid spatialGrids_;
};