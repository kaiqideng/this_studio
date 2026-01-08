#pragma once
#include "cudaKernel/SPHNeighborSearch.h"
#include "cudaKernel/SPHIntegration.h"
#include "cudaKernel/myStruct/particle.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "cudaKernel/myStruct/interaction.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"

class SPHHandler
{
public:
    SPHHandler()
    {
        downloadFlag_ = false;
    }

    ~SPHHandler() = default;

    void addSPHParticles(std::vector<double3> points, double3 velocity, double spacing, double density, double kinematicViscosity, cudaStream_t stream)
    {
        if(!downloadFlag_) 
        {
            downloadFlag_ = true;
            SPHAndGhosts_.upload(stream);
        }
        double mass = spacing * spacing * spacing * density;
        for (size_t i = 0; i < points.size(); i++)
        {
            SPHAndGhosts_.addSPHHost(points[i], velocity, 0.0, mass, density, 1.3 * spacing, kinematicViscosity);
        }
    }

    void addGhostParticles(std::vector<double3> points, double3 velocity, double spacing, double density, cudaStream_t stream)
    {
        if(!downloadFlag_) 
        {
            downloadFlag_ = true;
            SPHAndGhosts_.upload(stream);
        }
        double mass = spacing * spacing * spacing * density;
        for (size_t i = 0; i < points.size(); i++)
        {
            SPHAndGhosts_.addGhostHost(points[i], velocity, 0.0, mass, density, 1.3 * spacing, 0.0);
        }
    }

    void download(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if(downloadFlag_)
        {
            SPHAndGhosts_.download(stream);
            SPHInteractions_.alloc(SPHAndGhosts_.SPHDeviceSize() * 80, stream);
            SPHInteractionMap_.alloc(SPHAndGhosts_.SPHDeviceSize() + SPHAndGhosts_.ghostDeviceSize(),
            SPHAndGhosts_.SPHDeviceSize() + SPHAndGhosts_.ghostDeviceSize(), 
            stream);
            double cellSizeOneDim = 0.0;
            std::vector<double> h = SPHAndGhosts_.smoothLengthVector();
            if(h.size() > 0) cellSizeOneDim = *std::max_element(h.begin(), h.end()) * 2.0;
            spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream);
            downloadFlag_ = false;
        }
    }

    void neighborSearch(const size_t maxThreads, cudaStream_t stream)
    {
        launchSPHNeighborSearch(SPHInteractions_, 
        SPHInteractionMap_, 
        SPHAndGhosts_, 
        spatialGrids_, 
        maxThreads, 
        stream);
    }

    void integration1st(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchSPH1stIntegration(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        g,
        dt,
        maxThreads,
        stream);
    }

    void integration2nd(const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchSPH2ndIntegration(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        dt,
        maxThreads,
        stream);
    }

    void integration3rd(const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchSPH3rdIntegration(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        dt,
        maxThreads,
        stream);
    }

    void updateBoundaryCondition(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchAdamiBoundaryCondition(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        g,
        dt,
        maxThreads,
        stream);
    }

    SPH& getSPHAndGhosts() {return SPHAndGhosts_;}

    void outputSPHVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/SPHAndGhosts_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = SPHAndGhosts_.SPHDeviceSize() + SPHAndGhosts_.ghostDeviceSize();
        const std::vector<double3> p = SPHAndGhosts_.positionVector();
        const std::vector<double3> v = SPHAndGhosts_.velocityVector();
        const std::vector<double> h = SPHAndGhosts_.smoothLengthVector();
        const std::vector<double> pr = SPHAndGhosts_.pressureVector();
        
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

        out << "      <PointData Scalars=\"smoothLength\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"smoothLength\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << h[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << pr[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "velocity"       , v     }
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
    bool downloadFlag_;

    SPH SPHAndGhosts_;
    spatialGrid spatialGrids_;

    SPHInteraction SPHInteractions_;
    interactionMap SPHInteractionMap_;
};