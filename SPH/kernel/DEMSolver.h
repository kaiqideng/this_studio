#include "buildHashStartEnd.h"
#include "particle.h"
#include "wall.h"
#include "interaction.h"
#include "boundary.h"
#include "ballNeighborSearchKernel.h"
#include "ballIntegrationKernel.h"
#include "myUtility/myFileEdit.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

struct solidInteraction
{
    pair pair_;
    spring spring_;
    contact contact_;

    objectPointed objectPointed_;
    objectPointing objectPointing_;

    pair oldPair_;
    spring oldSpring_;

    void copyOld(cudaStream_t stream)
    {
        size_t numNeighborPairs = objectPointed_.numNeighborPairs();
        if (numNeighborPairs == 0) return;
        if (numNeighborPairs > oldPair_.deviceSize())
        {
            oldPair_.allocateDevice(numNeighborPairs, stream);
            oldSpring_.allocateDevice(numNeighborPairs, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.objectPointed(), pair_.objectPointed(), numNeighborPairs * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.objectPointing(), pair_.objectPointing(), numNeighborPairs * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.hashIndex(), pair_.hashIndex(), numNeighborPairs * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.sliding(), spring_.sliding(), numNeighborPairs * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.rolling(), spring_.rolling(), numNeighborPairs * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.torsion(), spring_.torsion(), numNeighborPairs * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
    }

    void resizePairSpringContact(cudaStream_t stream)
    {
        size_t numNeighborPairs = objectPointed_.numNeighborPairs();
        if (numNeighborPairs == 0) return;
        if (numNeighborPairs > pair_.deviceSize())
        {
            pair_.allocateDevice(numNeighborPairs, stream);
            spring_.allocateDevice(numNeighborPairs, stream);
            contact_.allocateDevice(numNeighborPairs, stream);
        }
    }

    void buildObjectPointingInteractionStartEnd(cudaStream_t stream)
    {
        size_t numNeighborPairs = objectPointed_.numNeighborPairs();
        if (numNeighborPairs == 0) return;
        size_t blockD = 256;
        if (numNeighborPairs < 256) blockD = numNeighborPairs;
        size_t gridD = (numNeighborPairs + blockD - 1) / blockD;
        buildHashStartEnd(objectPointing_.interactionStart(), 
        objectPointing_.interactionEnd(), 
        pair_.hashIndex(), 
        pair_.objectPointing(), 
        objectPointing_.deviceSize(), 
        numNeighborPairs, 
        gridD, 
        blockD, 
        stream);
    }
};

struct bondedInteraction
{
    pair pair_;
    bond bond_;
};

struct meshWall
{
    clump clump_;
    triangle triangle_;
    vertex vertex_;
};

class DEMSolver
{
public:
    DEMSolver(cudaStream_t s)
    {
        stream_ = s;
    }

    ~DEMSolver() = default;

protected:
    void neighborSearch()
    {
        updateSpatialGridCellHashStartEnd(ball_.position(), 
        ball_.hashIndex(), 
        ball_.hashValue(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.maximumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ballSpatialGrid_.numGrids(),
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

        ballAndBall_.copyOld(stream_);

        launchCountBallInteractions(ball_.position(), 
        ball_.radius(), 
        ball_.inverseMass(), 
        ball_.clumpID(), 
        ball_.hashIndex(), 
        ballAndBall_.objectPointed_.neighborCount(), 
        ballAndBall_.objectPointed_.neighborPrefixSum(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

        ballAndBall_.resizePairSpringContact(stream_);

        launchWriteBallInteractions(ball_.position(), 
        ball_.radius(), 
        ball_.inverseMass(),
        ball_.clumpID(),
        ball_.hashIndex(), 
        ballAndBall_.objectPointed_.neighborPrefixSum(),
        ballAndBall_.objectPointing_.interactionStart(),
        ballAndBall_.objectPointing_.interactionEnd(),
        ballAndBall_.spring_.sliding(),
        ballAndBall_.spring_.rolling(),
        ballAndBall_.spring_.torsion(),
        ballAndBall_.pair_.objectPointed(),
        ballAndBall_.pair_.objectPointing(),
        ballAndBall_.oldSpring_.sliding(),
        ballAndBall_.oldSpring_.rolling(),
        ballAndBall_.oldSpring_.torsion(),
        ballAndBall_.oldPair_.objectPointed(),
        ballAndBall_.oldPair_.hashIndex(),
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(),
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(),
        stream_);

        ballAndBall_.buildObjectPointingInteractionStartEnd(stream_);

        if (dummy_.deviceSize() == 0) return;

        updateSpatialGridCellHashStartEnd(dummy_.position(), 
        dummy_.hashIndex(), 
        dummy_.hashValue(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.maximumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ballSpatialGrid_.numGrids(),
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

        ballAndDummy_.copyOld(stream_);

        launchCountBallDummyInteractions(ball_.position(), 
        ball_.radius(), 
        ballAndDummy_.objectPointed_.neighborCount(),
        ballAndDummy_.objectPointed_.neighborPrefixSum(),
        dummy_.position(), 
        dummy_.radius(), 
        dummy_.hashIndex(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

        ballAndDummy_.resizePairSpringContact(stream_);

        launchWriteBallDummyInteractions(ball_.position(), 
        ball_.radius(), 
        ballAndDummy_.objectPointed_.neighborPrefixSum(),
        dummy_.position(), 
        dummy_.radius(),
        dummy_.hashIndex(),
        ballAndDummy_.objectPointing_.interactionStart(),
        ballAndDummy_.objectPointing_.interactionEnd(),
        ballAndDummy_.spring_.sliding(),
        ballAndDummy_.spring_.rolling(),
        ballAndDummy_.spring_.torsion(),
        ballAndDummy_.pair_.objectPointed(),
        ballAndDummy_.pair_.objectPointing(),
        ballAndDummy_.oldSpring_.sliding(),
        ballAndDummy_.oldSpring_.rolling(),
        ballAndDummy_.oldSpring_.torsion(),
        ballAndDummy_.oldPair_.objectPointed(),
        ballAndDummy_.oldPair_.hashIndex(),
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(),
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

        ballAndDummy_.buildObjectPointingInteractionStartEnd(stream_);
    }

    void triangleNeighborSearch()
    {
        updateSpatialGridCellHashStartEnd(meshWall_.triangle_.circumcenter(), 
        meshWall_.triangle_.hashIndex(), 
        meshWall_.triangle_.hashValue(), 
        triangleSpatialGrid_.cellHashStart(), 
        triangleSpatialGrid_.cellHashEnd(), 
        triangleSpatialGrid_.minimumBoundary(), 
        triangleSpatialGrid_.maximumBoundary(), 
        triangleSpatialGrid_.cellSize(), 
        triangleSpatialGrid_.gridSize(), 
        triangleSpatialGrid_.numGrids(),
        meshWall_.triangle_.deviceSize(), 
        meshWall_.triangle_.gridDim(), 
        meshWall_.triangle_.blockDim(), 
        stream_);

        ballAndTriangle_.copyOld(stream_);

        launchCountBallTriangleInteractions(ball_.position(), 
        ball_.radius(), 
        ballAndTriangle_.objectPointed_.neighborCount(),
        ballAndTriangle_.objectPointed_.neighborPrefixSum(),
        meshWall_.triangle_.index0(), 
        meshWall_.triangle_.index1(), 
        meshWall_.triangle_.index2(),
        meshWall_.triangle_.hashIndex(), 
        meshWall_.vertex_.globalPosition(),
        triangleSpatialGrid_.cellHashStart(), 
        triangleSpatialGrid_.cellHashEnd(), 
        triangleSpatialGrid_.minimumBoundary(), 
        triangleSpatialGrid_.cellSize(), 
        triangleSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

        ballAndTriangle_.resizePairSpringContact(stream_);

        launchWriteBallTriangleInteractions(ball_.position(), 
        ball_.radius(), 
        ballAndTriangle_.objectPointed_.neighborPrefixSum(),
        meshWall_.triangle_.index0(), 
        meshWall_.triangle_.index1(), 
        meshWall_.triangle_.index2(),
        meshWall_.triangle_.hashIndex(),
        ballAndTriangle_.objectPointing_.interactionStart(),
        ballAndTriangle_.objectPointing_.interactionEnd(),
        meshWall_.vertex_.globalPosition(),
        ballAndTriangle_.spring_.sliding(),
        ballAndTriangle_.spring_.rolling(),
        ballAndTriangle_.spring_.torsion(),
        ballAndTriangle_.pair_.objectPointed(),
        ballAndTriangle_.pair_.objectPointing(),
        ballAndTriangle_.oldSpring_.sliding(),
        ballAndTriangle_.oldSpring_.rolling(),
        ballAndTriangle_.oldSpring_.torsion(),
        ballAndTriangle_.oldPair_.objectPointed(),
        ballAndTriangle_.oldPair_.hashIndex(),
        triangleSpatialGrid_.cellHashStart(), 
        triangleSpatialGrid_.cellHashEnd(),
        triangleSpatialGrid_.minimumBoundary(),
        triangleSpatialGrid_.cellSize(),
        triangleSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

        ballAndTriangle_.buildObjectPointingInteractionStartEnd(stream_);       
    }

    void interaction1stHalf(const double3 gravity, const double timeStep)
    {
        launchClump1stHalfIntegration(clump_.position(),
        clump_.velocity(),
        clump_.angularVelocity(),
        clump_.force(),
        clump_.torque(),
        clump_.inverseMass(),
        clump_.orientation(),
        clump_.inverseInertiaTensor(),
        clump_.pebbleStart(),
        clump_.pebbleEnd(),
        ball_.position(),
        ball_.velocity(),
        ball_.angularVelocity(),
        gravity,
        timeStep,
        clump_.deviceSize(),
        clump_.gridDim(),
        clump_.blockDim(),
        stream_);

        launchBall1stHalfIntegration(ball_.position(),
        ball_.velocity(),
        ball_.angularVelocity(),
        ball_.force(),
        ball_.torque(),
        ball_.radius(),
        ball_.inverseMass(),
        ball_.clumpID(),
        gravity,
        timeStep,
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);
    }

    void interaction2ndHalf(const double3 gravity, const double timeStep)
    {
        launchClump2ndHalfIntegration(clump_.position(),
        clump_.velocity(),
        clump_.angularVelocity(),
        clump_.force(),
        clump_.torque(),
        clump_.inverseMass(),
        clump_.orientation(),
        clump_.inverseInertiaTensor(),
        clump_.pebbleStart(),
        clump_.pebbleEnd(),
        ball_.position(),
        ball_.velocity(),
        ball_.angularVelocity(),
        ball_.force(),
        ball_.torque(),
        gravity,
        timeStep,
        clump_.deviceSize(),
        clump_.gridDim(),
        clump_.blockDim(),
        stream_);

        launchBall2ndHalfIntegration(ball_.velocity(),
        ball_.angularVelocity(),
        ball_.force(),
        ball_.torque(),
        ball_.radius(),
        ball_.inverseMass(),
        ball_.clumpID(),
        gravity,
        timeStep,
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);
    }

    void outputBallVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/ball_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = ball_.deviceSize();
        std::vector<double3> p = ball_.positionHostCopy();
        std::vector<double3> v = ball_.velocityHostCopy();
        std::vector<double3> a = ball_.angularVelocityHostCopy();
        const std::vector<double> r = ball_.radiusHostRef();
        const std::vector<int> materialID = ball_.materialIDHostRef();
        const std::vector<int> clumpID = ball_.clumpIDHostRef();
        
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

    void outputMeshWallVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/triangleWall_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        std::ofstream out(fname.str().c_str());
        if (!out) {
            throw std::runtime_error("Cannot open " + fname.str());
        }

        out << std::fixed << std::setprecision(10);

        std::vector<double3> verts = meshWall_.vertex_.globalPositionHostCopy();
        const size_t numPoints = verts.size();

        const std::vector<int> tri_i0 = meshWall_.triangle_.index0HostRef();
        const std::vector<int> tri_i1 = meshWall_.triangle_.index1HostRef();
        const std::vector<int> tri_i2 = meshWall_.triangle_.index2HostRef();
        const size_t numTris = tri_i0.size();

        const std::vector<int> triWallId = meshWall_.triangle_.wallIndexHostRef();

        std::vector<int> triMaterialId(numTris, 0);
        const std::vector<int> wallMaterial = meshWall_.clump_.materialIDHostRef();
        if (!wallMaterial.empty() && !triWallId.empty()) {
            for (size_t t = 0; t < numTris; ++t) {
                const int w = triWallId[t];
                if (w >= 0 && static_cast<size_t>(w) < wallMaterial.size()) {
                    triMaterialId[t] = wallMaterial[w];
                }
            }
        }

        out << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            << "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            << "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            << "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            << "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << numPoints
            << "\" NumberOfCells=\"" << numTris << "\">\n";

        out << "      <Points>\n"
            << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < numPoints; ++i) {
            out << ' ' << verts[i].x
                << ' ' << verts[i].y
                << ' ' << verts[i].z;
        }
        out << "\n        </DataArray>\n"
            << "      </Points>\n";

        out << "      <Cells>\n";

        out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << tri_i0[t]
                << ' ' << tri_i1[t]
                << ' ' << tri_i2[t];
        }
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (size_t t = 1; t <= numTris; ++t) {
            out << ' ' << (3 * t);
        }
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << " 5";
        }
        out << "\n        </DataArray>\n"
            << "      </Cells>\n";

        out << "      <CellData Scalars=\"materialID\">\n";

        if (!triWallId.empty()) {
            out << "        <DataArray type=\"Int32\" Name=\"wallID\" format=\"ascii\">\n";
            for (size_t t = 0; t < numTris; ++t) {
                out << ' ' << triWallId[t];
            }
            out << "\n        </DataArray>\n";
        }

        out << "        <DataArray type=\"Int32\" Name=\"materialID\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << triMaterialId[t];
        }
        out << "\n        </DataArray>\n";

        out << "      </CellData>\n";

        out << "    </Piece>\n"
            << "  </UnstructuredGrid>\n"
            << "</VTKFile>\n";
    }

private:
    cudaStream_t stream_;

    ball ball_;
    ball dummy_;
    clump clump_;
    meshWall meshWall_;

    solidInteraction ballAndBall_;
    bondedInteraction bondedBallAndBall_;
    solidInteraction ballAndDummy_;
    solidInteraction ballAndTriangle_;

    spatialGrid ballSpatialGrid_;
    spatialGrid triangleSpatialGrid_;
};