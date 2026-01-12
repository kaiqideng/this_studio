#pragma once
#include "SPH.h"
#include "SPHNeighborSearch.h"
#include "SPHIntegration.h"
#include "DEM/kernel/cudaKernel/myStruct/spatialGrid.h"
#include "DEM/kernel/cudaKernel/myStruct/myUtility/myFileEdit.h"

class WCSPHHandler
{
public:
    WCSPHHandler()
    {
        downloadFlag_ = false;
    }

    ~WCSPHHandler() = default;

    void addWCSPHParticles(std::vector<double3> points, double3 velocity, double soundSpeed, double spacing, double density, double kinematicViscosity, cudaStream_t stream)
    {
        if(!downloadFlag_) 
        {
            downloadFlag_ = true;
            WCSPHs_.upload(stream);
        }
        double mass = spacing * spacing * spacing * density;
        for (size_t i = 0; i < points.size(); i++)
        {
            WCSPHs_.addSPHHost(points[i], velocity, density, 0.0, soundSpeed, mass, density, 1.3 * spacing, kinematicViscosity);
        }
    }

    void addWCSPHDummyParticles(std::vector<double3> points, double3 velocity, double soundSpeed, double spacing, double density, cudaStream_t stream)
    {
        if(!downloadFlag_) 
        {
            downloadFlag_ = true;
            WCSPHs_.upload(stream);
        }
        double mass = spacing * spacing * spacing * density;
        for (size_t i = 0; i < points.size(); i++)
        {
            WCSPHs_.addDummyHost(points[i], velocity, density, 0.0, soundSpeed, mass, density, 1.3 * spacing, 0.0);
        }
    }

    void download(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if (downloadFlag_)
        {
            WCSPHs_.download(stream);
            SPHInteractions_.alloc(WCSPHs_.SPHDeviceSize() * 80, stream);
            SPHInteractionMap_.alloc(WCSPHs_.SPHDeviceSize() + WCSPHs_.dummyDeviceSize(),
            WCSPHs_.SPHDeviceSize() + WCSPHs_.dummyDeviceSize(), 
            stream);
            double cellSizeOneDim = 0.0;
            std::vector<double> h = WCSPHs_.smoothLengthVector();
            if (h.size() > 0) cellSizeOneDim = *std::max_element(h.begin(), h.end()) * 2.0;
            spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream);
            downloadFlag_ = false;
        }
    }

    void WCSPHNeighborSearch(const size_t maxThreads, cudaStream_t stream)
    {
        launchSPHNeighborSearch(SPHInteractions_, 
        SPHInteractionMap_, 
        spatialGrids_, 
        WCSPHs_.hashIndex(),
        WCSPHs_.hashValue(),
        WCSPHs_.position(),
        WCSPHs_.smoothLength(),
        WCSPHs_.SPHDeviceSize(),
        WCSPHs_.dummyDeviceSize(),
        maxThreads,
        stream);
    }

    void WCSPH1stIntegration(const double3 g, const double dt, const size_t gridDim_SPH, const size_t blockDim_SPH, cudaStream_t stream)
    {
        launchWCSPH1stIntegration(WCSPHs_,
        SPHInteractions_,
        SPHInteractionMap_,
        g,
        dt,
        gridDim_SPH,
        blockDim_SPH,
        stream);
    }

    void WCSPH2ndIntegration(const double3 g, const double dt, const size_t gridDim_SPH, const size_t blockDim_SPH, cudaStream_t stream)
    {
        launchWCSPH2ndIntegration(WCSPHs_,
        SPHInteractions_,
        SPHInteractionMap_,
        g,
        dt,
        gridDim_SPH,
        blockDim_SPH,
        stream);
    }

    void setDummyParticleBoundary(const size_t maxThreads, cudaStream_t stream)
    {
        launchConfigDummyParticles(WCSPHs_, 
		SPHInteractions_, 
		SPHInteractionMap_, 
		maxThreads, 
		stream);
    }

    WCSPH& getWCSPHs() {return WCSPHs_;}

    void outputWCSPHVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/WCSPHs_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const std::vector<double3> p = WCSPHs_.positionVector();
        const std::vector<double3> v = WCSPHs_.velocityVector();
        const std::vector<double> h = WCSPHs_.smoothLengthVector();
        const std::vector<double> pr = WCSPHs_.pressureVector();
        size_t N = WCSPHs_.SPHDeviceSize() + WCSPHs_.dummyDeviceSize();
        
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

    WCSPH WCSPHs_;
    spatialGrid spatialGrids_;

    SPHInteraction SPHInteractions_;
    interactionMap SPHInteractionMap_;
};