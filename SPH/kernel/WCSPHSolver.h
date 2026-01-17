#include "particle.h"
#include "interaction.h"
#include "boundary.h"
#include "SPHNeighborSearchKernel.h"
#include "WCSPHIntegrationKernel.h"
#include "myUtility/myFileEdit.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

struct SPHInteraction
{
    objectPointed objectPointed_;
    interaction interaction_;
};

class WCSPHSolver
{
public:
    WCSPHSolver(cudaStream_t s)
    {
        stream_ = s;
    }

    ~WCSPHSolver() = default;

protected:
    void initializeBoundaryCondition()
    {
        launchCountSPHInteractions(dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.hashIndex(), 
        dummy_.hashValue(), 
        dummyAndDummy_.objectPointed_.neighborCount(), 
        dummyAndDummy_.objectPointed_.neighborPrefixSum(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(), 
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

        size_t numNeighborPairs = dummyAndDummy_.objectPointed_.numNeighborPairs();
        if (numNeighborPairs > dummyAndDummy_.interaction_.deviceSize())
        {
            dummyAndDummy_.interaction_.allocateDevice(numNeighborPairs, stream_);
        }

        launchWriteSPHInteractions(dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.hashIndex(), 
        dummyAndDummy_.objectPointed_.neighborPrefixSum(),
        dummyAndDummy_.interaction_.objectPointed(),
        dummyAndDummy_.interaction_.objectPointing(),
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(),
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

        launchCalDummyParticleNormal(dummy_.normal(), 
        dummy_.position(), 
        dummy_.density(), 
        dummy_.mass(), 
        dummy_.smoothLength(), 
        dummyAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummyAndDummy_.interaction_.objectPointing(), 
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);
    }

    void upload()
    {
        WCSPH_.copyHostToDevice(stream_);
        dummy_.copyHostToDevice(stream_);

        size_t numSPH = WCSPH_.deviceSize();
        SPHAndSPH_.objectPointed_.allocateDevice(numSPH, stream_);
        SPHAndSPH_.interaction_.allocateDevice(80 * numSPH, stream_);
        SPHAndDummy_.objectPointed_.allocateDevice(numSPH, stream_);
        SPHAndDummy_.interaction_.allocateDevice(80 * numSPH, stream_);
        size_t numDummy = dummy_.deviceSize();
        dummyAndDummy_.objectPointed_.allocateDevice(numDummy, stream_);
        dummyAndDummy_.interaction_.allocateDevice(80 * numDummy, stream_);
    }

    void initialize(const double3 minBoundary, const double3 maxBoundary, const size_t maximumThread = 256)
    {
        WCSPH_.setBlockDim(WCSPH_.hostSize() < maximumThread ? WCSPH_.hostSize() : maximumThread);
        dummy_.setBlockDim(dummy_.hostSize() < maximumThread ? dummy_.hostSize() : maximumThread);

        upload();

        double cellSizeOneDim = 0.0;
        std::vector<double> h = WCSPH_.smoothLengthHostCopy();
        if (h.size() > 0) cellSizeOneDim = *std::max_element(h.begin(), h.end()) * 2.0;
        std::vector<double> h1 = dummy_.smoothLengthHostCopy();
        if (h1.size() > 0) cellSizeOneDim = std::max(cellSizeOneDim, *std::max_element(h1.begin(), h1.end()) * 2.0);
        spatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);

        initializeBoundaryCondition();
    }

    void neighborSearch()
    {
        launchCountSPHInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        WCSPH_.hashIndex(), 
        WCSPH_.hashValue(), 
        SPHAndSPH_.objectPointed_.neighborCount(), 
        SPHAndSPH_.objectPointed_.neighborPrefixSum(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);

        size_t numNeighborPairs = SPHAndSPH_.objectPointed_.numNeighborPairs();
        if (numNeighborPairs > SPHAndSPH_.interaction_.deviceSize())
        {
            SPHAndSPH_.interaction_.allocateDevice(numNeighborPairs, stream_);
        }

        launchWriteSPHInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        WCSPH_.hashIndex(), 
        SPHAndSPH_.objectPointed_.neighborPrefixSum(),
        SPHAndSPH_.interaction_.objectPointed(),
        SPHAndSPH_.interaction_.objectPointing(),
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(),
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);

        if (dummy_.deviceSize() == 0) return;

        launchCountSPHDummyInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        SPHAndDummy_.objectPointed_.neighborCount(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.hashIndex(), 
        dummy_.hashValue(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(),
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

        numNeighborPairs = SPHAndDummy_.objectPointed_.numNeighborPairs();
        if (numNeighborPairs > SPHAndDummy_.interaction_.deviceSize())
        {
            SPHAndDummy_.interaction_.allocateDevice(numNeighborPairs, stream_);
        }

        launchWriteSPHDummyInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.hashIndex(), 
        SPHAndDummy_.interaction_.objectPointed(),
        SPHAndDummy_.interaction_.objectPointing(),
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(),
        stream_);
    }

    void interaction1stHalf(const double3 gravity, const double timeStep)
    {
        launchWCSPH1stHalfIntegration(WCSPH_.position(), 
        WCSPH_.velocity(), 
        WCSPH_.density(), 
        WCSPH_.pressure(), 
        WCSPH_.soundSpeed(), 
        WCSPH_.mass(), 
        WCSPH_.initialDensity(), 
        WCSPH_.smoothLength(), 
        SPHAndSPH_.objectPointed_.neighborPrefixSum(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.velocity(), 
        dummy_.normal(), 
        dummy_.soundSpeed(), 
        dummy_.mass(), 
        dummy_.initialDensity(), 
        dummy_.smoothLength(), 
        SPHAndSPH_.interaction_.objectPointing(), 
        SPHAndDummy_.interaction_.objectPointing(), 
        gravity, 
        timeStep, 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(),
        stream_);
    }

    void interaction2ndHalf(const double3 gravity, const double timeStep)
    {
        launchWCSPH2ndHalfIntegration(WCSPH_.position(), 
        WCSPH_.velocity(), 
        WCSPH_.density(), 
        WCSPH_.pressure(), 
        WCSPH_.soundSpeed(), 
        WCSPH_.mass(), 
        WCSPH_.initialDensity(), 
        WCSPH_.smoothLength(), 
        SPHAndSPH_.objectPointed_.neighborPrefixSum(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.velocity(), 
        dummy_.normal(), 
        dummy_.soundSpeed(), 
        dummy_.mass(), 
        dummy_.initialDensity(), 
        dummy_.smoothLength(), 
        SPHAndSPH_.interaction_.objectPointing(), 
        SPHAndDummy_.interaction_.objectPointing(), 
        gravity, 
        timeStep, 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);
    }

    void outputWCSPHVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/SPH_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        size_t N = WCSPH_.hostSize();
        std::vector<double3> p = WCSPH_.positionHostCopy();
        std::vector<double3> v = WCSPH_.velocityHostCopy();
        std::vector<double> pr = WCSPH_.pressureHostCopy();
        std::vector<double> h = WCSPH_.smoothLengthHostCopy();
        
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

    void outputDummyVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/dummy_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        size_t N = dummy_.hostSize();
        std::vector<double3> p = dummy_.positionHostCopy();
        std::vector<double3> v = dummy_.velocityHostCopy();
        std::vector<double> h = dummy_.smoothLengthHostCopy();
        
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

public:
    void addSPH(double3 position, double3 velocity, double soundSpeed, double spacing, double density, double viscosity)
    {
        double mass = spacing * spacing * spacing * density;
        WCSPH_.addHost(position, velocity, density, 0.0, density, 1.3 * spacing, mass, soundSpeed, viscosity);
    }

    void addDummy(double3 position, double3 velocity, double soundSpeed, double spacing, double density, double viscosity)
    {
        double mass = spacing * spacing * spacing * density;
        dummy_.addHost(position, velocity, density, 0.0, density, 1.3 * spacing, mass, soundSpeed, viscosity);
    }

    void eraseSPH(std::vector<size_t> index)
    {
        WCSPH_.eraseHost(index);
    }

    void eraseDummy(std::vector<size_t> index)
    {
        dummy_.eraseHost(index);
    }

    void copySPH(const WCSPH& other)
    {
        WCSPH_.copyFromHost(other);
    }

    void copyDummy(const WCSPH& other)
    {
        dummy_.copyFromHost(other);
    }

    const WCSPH& getSPH() const { return WCSPH_; }

    const WCSPH& getDummy() const { return dummy_; }

    void solve(const double3 minBoundary, 
    const double3 maxBoundary, 
    const double3 gravity, 
    const double timeStep, 
    const double maximumTime,
    const size_t numFrame,
    const std::string dir, 
    const size_t deviceID = 0, 
    const size_t maximumThread = 256)
    {
        removeVtuFiles(dir);
        removeDatFiles(dir);

        cudaError_t cudaStatus = cudaSetDevice(deviceID);
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "cudaSetDevice( " << deviceID
            << " ) failed! Do you have a CUDA-capable GPU installed?"
            << std::endl;
            exit(1);
        }

        size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        
        initialize(minBoundary, maxBoundary, maximumThread);
        std::cout << "Initialization Completed." << std::endl;

        outputWCSPHVTU(dir, 0, 0, 0.0);
        outputDummyVTU(dir, 0, 0, 0.0);

        size_t iStep = 0, iFrame = 0;
        double time = 0.0;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            neighborSearch();
            interaction1stHalf(gravity, timeStep);
            interaction2ndHalf(gravity, timeStep);
            if (iStep % frameInterval == 0) 
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputWCSPHVTU(dir, iFrame, iStep, time);
                outputDummyVTU(dir, iFrame, iStep, time);
            }
        }
        WCSPH_.copyDeviceToHost(stream_);
        dummy_.copyDeviceToHost(stream_);
    }

private:
    cudaStream_t stream_;

    WCSPH WCSPH_;
    WCSPH dummy_;

    SPHInteraction SPHAndSPH_;
    SPHInteraction SPHAndDummy_;
    SPHInteraction dummyAndDummy_;
    spatialGrid spatialGrid_;
};