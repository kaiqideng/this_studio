#include "buildHashStartEnd.h"
#include "contactParameters.h"
#include "myUtility/myMat.h"
#include "particle.h"
#include "wall.h"
#include "interaction.h"
#include "boundary.h"
#include "ballNeighborSearchKernel.h"
#include "contactKernel.h"
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

    size_t numActivated_ { 0 };

    void updateOldPairOldSpring(cudaStream_t stream)
    {
        if (numActivated_ == 0) return;
        if (numActivated_ > oldPair_.deviceSize())
        {
            oldPair_.allocateDevice(numActivated_, stream);
            oldSpring_.allocateDevice(numActivated_, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.objectPointed(), pair_.objectPointed(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.objectPointing(), pair_.objectPointing(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.hashIndex(), pair_.hashIndex(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.hashValue(), pair_.hashValue(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.cancelFlag(), pair_.cancelFlag(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.sliding(), spring_.sliding(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.rolling(), spring_.rolling(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.torsion(), spring_.torsion(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
    }

    void resizePair(cudaStream_t stream)
    {
        numActivated_ = objectPointed_.numNeighborPairs();
        if (numActivated_ > pair_.deviceSize())
        {
            pair_.allocateDevice(numActivated_, stream);
            spring_.allocateDevice(numActivated_, stream);
            contact_.allocateDevice(numActivated_, stream);
        }
    }

    void buildObjectPointingInteractionStartEnd(size_t maxThread, cudaStream_t stream)
    {
        if (numActivated_ == 0) return;
        size_t blockD = maxThread;
        if (numActivated_ < maxThread) blockD = numActivated_;
        size_t gridD = (numActivated_ + blockD - 1) / blockD;
        CUDA_CHECK(cudaMemcpyAsync(pair_.hashValue(), pair_.objectPointing(), pair_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        buildHashStartEnd(objectPointing_.interactionStart(), 
        objectPointing_.interactionEnd(), 
        pair_.hashIndex(), 
        pair_.hashValue(), 
        objectPointing_.deviceSize(), 
        numActivated_, 
        gridD, 
        blockD, 
        stream);
    }
};

struct bondedInteraction
{
    pair pair_;
    bond bond_;

    void add(const std::vector<int>& ob0,
    const std::vector<int>& ob1,
    const std::vector<double3>& p)
    {
        if (ob0.size() != ob1.size()) return;

        std::vector<int> existingPointed  = pair_.objectPointedHostCopy();
        std::vector<int> existingPointing = pair_.objectPointingHostCopy();

        for (size_t i = 0; i < ob0.size(); ++i)
        {
            int i0 = ob0[i];
            int i1 = ob1[i];

            if (i0 < 0 || i1 < 0) continue;
            if (static_cast<size_t>(i0) >= p.size() || static_cast<size_t>(i1) >= p.size()) continue;
            if (i0 == i1) continue;

            int a = i0;
            int b = i1;
            if (a > b) std::swap(a, b);

            bool found = false;
            for (size_t j = 0; j < existingPointed.size(); ++j)
            {
                if (existingPointed[j] == a && existingPointing[j] == b)
                {
                    found = true;
                    break;
                }
            }
            if (found) continue;

            existingPointed.push_back(a);
            existingPointing.push_back(b);

            pair_.addHost(a, b);
            double3 n = p[a] - p[b];
            double3 n_norm = normalize(n);
            bond_.addHost(0.5 * (p[a] + p[b]), n_norm);
        }
    }
};

struct meshWall
{
    clump body_;
    triangle triangle_;
    vertex vertex_;
};

class DEMSolver
{
public:
    DEMSolver(cudaStream_t s)
    {
        maxThread_ = 256;
        stream_ = s;
    }

    ~DEMSolver() = default;

private:
    void upload()
    {
        contactModelParameters_.buildFromTables(hertzianTable_, 
        linearTable_, 
        bondedTable_, 
        stream_);

        ball_.copyHostToDevice(stream_);
        dummy_.copyHostToDevice(stream_);
        clump_.copyHostToDevice(stream_);
        meshWall_.body_.copyHostToDevice(stream_);
        meshWall_.triangle_.copyHostToDevice(stream_);
        meshWall_.vertex_.copyHostToDevice(stream_);

        ballAndBall_.pair_.allocateDevice(6 * ball_.deviceSize(), stream_);
        ballAndBall_.spring_.allocateDevice(6 * ball_.deviceSize(), stream_);
        ballAndBall_.contact_.allocateDevice(6 * ball_.deviceSize(), stream_);
        ballAndBall_.oldPair_.allocateDevice(6 * ball_.deviceSize(), stream_);
        ballAndBall_.oldSpring_.allocateDevice(6 * ball_.deviceSize(), stream_);
        ballAndBall_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
        ballAndBall_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);

        if (dummy_.deviceSize() > 0)
        {
            ballAndDummy_.pair_.allocateDevice(6 * ball_.deviceSize(), stream_);
            ballAndDummy_.spring_.allocateDevice(6 * ball_.deviceSize(), stream_);
            ballAndDummy_.contact_.allocateDevice(6 * ball_.deviceSize(), stream_);
            ballAndDummy_.oldPair_.allocateDevice(6 * ball_.deviceSize(), stream_);
            ballAndDummy_.oldSpring_.allocateDevice(6 * ball_.deviceSize(), stream_);
            ballAndDummy_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            ballAndDummy_.objectPointing_.allocateDevice(dummy_.deviceSize(), stream_);
        }

        if (meshWall_.triangle_.deviceSize() > 0)
        {
            ballAndTriangle_.pair_.allocateDevice(3 * ball_.deviceSize(), stream_);
            ballAndTriangle_.spring_.allocateDevice(3 * ball_.deviceSize(), stream_);
            ballAndTriangle_.contact_.allocateDevice(3 *  ball_.deviceSize(), stream_);
            ballAndTriangle_.oldPair_.allocateDevice(3 * ball_.deviceSize(), stream_);
            ballAndTriangle_.oldSpring_.allocateDevice(3 * ball_.deviceSize(), stream_);
            ballAndTriangle_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            ballAndTriangle_.objectPointing_.allocateDevice(meshWall_.triangle_.deviceSize(), stream_);
        }

        bondedBallAndBall_.pair_.copyHostToDevice(stream_);
        bondedBallAndBall_.bond_.copyHostToDevice(stream_);
    }

    void initializeSpatialGrid(const double3 minBoundary, const double3 maxBoundary)
    {
        double cellSizeOneDim = 0.0;
        const std::vector<double> r = ball_.radiusHostRef();
        if (r.size() > 0) cellSizeOneDim = *std::max_element(r.begin(), r.end()) * 2.0;
        const std::vector<double> r1 = dummy_.radiusHostRef();
        if (r1.size() > 0) cellSizeOneDim = std::max(cellSizeOneDim, *std::max_element(r1.begin(), r1.end()) * 2.0);
        ballSpatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);

        double longestEdge = 0.0;
        for (size_t i = 0; i < meshWall_.triangle_.deviceSize(); i++)
        {
            double3 v0 = meshWall_.vertex_.localPositionHostRef()[meshWall_.triangle_.index0HostRef()[i]];
            double3 v1 = meshWall_.vertex_.localPositionHostRef()[meshWall_.triangle_.index1HostRef()[i]];
            double3 v2 = meshWall_.vertex_.localPositionHostRef()[meshWall_.triangle_.index2HostRef()[i]];
            longestEdge = std::max(longestEdge, length(v1 - v0));
            longestEdge = std::max(longestEdge, length(v2 - v1));
            longestEdge = std::max(longestEdge, length(v2 - v0));
        }
        cellSizeOneDim = std::max(cellSizeOneDim, longestEdge);
        triangleSpatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);
    }

    void initialize(const double3 minBoundary, const double3 maxBoundary, const size_t maximumThread)
    {
        maxThread_ = maximumThread;
        ball_.setBlockDim(maxThread_);
        dummy_.setBlockDim(maxThread_);
        clump_.setBlockDim(maxThread_);
        meshWall_.body_.setBlockDim(maxThread_);
        meshWall_.triangle_.setBlockDim(maxThread_);
        meshWall_.vertex_.setBlockDim(maxThread_);

        upload();

        initializeSpatialGrid(minBoundary, maxBoundary);
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

        ballAndTriangle_.updateOldPairOldSpring(stream_);

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

        ballAndTriangle_.resizePair(stream_);

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

        ballAndTriangle_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);
    }

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

        ballAndBall_.updateOldPairOldSpring(stream_);

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

        ballAndBall_.resizePair(stream_);

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

        ballAndBall_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);

        if (meshWall_.triangle_.deviceSize() > 0) triangleNeighborSearch();

        if (dummy_.deviceSize() > 0)
        {
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

            ballAndDummy_.updateOldPairOldSpring(stream_);

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

            ballAndDummy_.resizePair(stream_);

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

            ballAndDummy_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);
        }
    }

    void calculateContactForceTorque(const double timeStep)
    {
        if (ballAndBall_.numActivated_ > 0) 
        {
            size_t blockD = maxThread_;
            if (ballAndBall_.numActivated_  < maxThread_) blockD = ballAndBall_.numActivated_ ;
            size_t gridD = (ballAndBall_.numActivated_  + blockD - 1) / blockD;
            luanchCalculateBallContactForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.radius(),
            ball_.inverseMass(),
            ball_.materialID(),
            ballAndBall_.spring_.sliding(),
            ballAndBall_.spring_.rolling(),
            ballAndBall_.spring_.torsion(),
            ballAndBall_.contact_.force(),
            ballAndBall_.contact_.torque(),
            ballAndBall_.contact_.point(),
            ballAndBall_.contact_.normal(),
            ballAndBall_.contact_.overlap(),
            ballAndBall_.pair_.objectPointed(),
            ballAndBall_.pair_.objectPointing(),
            timeStep,
            ballAndBall_.numActivated_ ,
            gridD,
            blockD,
            stream_);
        }

        size_t numBondedPairs = bondedBallAndBall_.bond_.deviceSize();
        if (numBondedPairs > 0)
        {
            size_t blockD1 = maxThread_;
            if (numBondedPairs < maxThread_) blockD1 = numBondedPairs;
            size_t gridD1 = (numBondedPairs + blockD1 - 1) / blockD1;
            luanchCalculateBondedForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.force(),
            ball_.torque(),
            ball_.radius(),
            ball_.materialID(),
            ballAndBall_.objectPointed_.neighborPrefixSum(),
            ballAndBall_.contact_.force(),
            ballAndBall_.contact_.torque(),
            ballAndBall_.contact_.point(),
            ballAndBall_.contact_.normal(),
            ballAndBall_.pair_.objectPointing(),
            bondedBallAndBall_.bond_.point(),
            bondedBallAndBall_.bond_.normal(),
            bondedBallAndBall_.bond_.shearForce(),
            bondedBallAndBall_.bond_.bendingTorque(),
            bondedBallAndBall_.bond_.normalForce(),
            bondedBallAndBall_.bond_.torsionTorque(),
            bondedBallAndBall_.bond_.isBonded(),
            bondedBallAndBall_.pair_.objectPointed(),
            bondedBallAndBall_.pair_.objectPointing(),
            timeStep,
            numBondedPairs,
            gridD1,
            blockD1,
            stream_);
        }

        luanchSumBallContactForceTorque(ball_.position(),
        ball_.force(),
        ball_.torque(),
        ballAndBall_.objectPointed_.neighborPrefixSum(),
        ballAndBall_.objectPointing_.interactionStart(),
        ballAndBall_.objectPointing_.interactionEnd(),
        ballAndBall_.contact_.force(),
        ballAndBall_.contact_.torque(),
        ballAndBall_.contact_.point(),
        ballAndBall_.pair_.hashIndex(),
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

        if (ballAndTriangle_.numActivated_ > 0)
        {            
            luanchCalculateBallWallContactForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.force(),
            ball_.torque(),
            ball_.radius(),
            ball_.inverseMass(),
            ball_.materialID(),
            ballAndTriangle_.objectPointed_.neighborPrefixSum(),
            meshWall_.body_.position(),
            meshWall_.body_.velocity(),
            meshWall_.body_.angularVelocity(),
            meshWall_.body_.materialID(),
            meshWall_.triangle_.index0(),
            meshWall_.triangle_.index1(),
            meshWall_.triangle_.index2(),
            meshWall_.triangle_.wallIndex(),
            meshWall_.vertex_.globalPosition(),
            ballAndTriangle_.spring_.sliding(),
            ballAndTriangle_.spring_.rolling(),
            ballAndTriangle_.spring_.torsion(),
            ballAndTriangle_.contact_.force(),
            ballAndTriangle_.contact_.torque(),
            ballAndTriangle_.contact_.point(),
            ballAndTriangle_.contact_.normal(),
            ballAndTriangle_.contact_.overlap(),
            ballAndTriangle_.pair_.objectPointed(),
            ballAndTriangle_.pair_.objectPointing(),
            ballAndTriangle_.pair_.cancelFlag(),
            timeStep,
            ball_.deviceSize(),
            ball_.gridDim(),
            ball_.blockDim(),
            stream_);
        }

        if (ballAndDummy_.numActivated_ > 0) 
        {
            size_t blockD = maxThread_;
            if (ballAndDummy_.numActivated_ < maxThread_) blockD = ballAndDummy_.numActivated_;
            size_t gridD = (ballAndDummy_.numActivated_ + blockD - 1) / blockD;
            luanchCalculateBallDummyContactForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.radius(),
            ball_.inverseMass(),
            ball_.materialID(),
            dummy_.position(),
            dummy_.velocity(),
            dummy_.angularVelocity(),
            dummy_.radius(),
            dummy_.inverseMass(),
            dummy_.materialID(),
            ballAndDummy_.spring_.sliding(),
            ballAndDummy_.spring_.rolling(),
            ballAndDummy_.spring_.torsion(),
            ballAndDummy_.contact_.force(),
            ballAndDummy_.contact_.torque(),
            ballAndDummy_.contact_.point(),
            ballAndDummy_.contact_.normal(),
            ballAndDummy_.contact_.overlap(),
            ballAndDummy_.pair_.objectPointed(),
            ballAndDummy_.pair_.objectPointing(),
            timeStep,
            ballAndDummy_.numActivated_,
            gridD,
            blockD,
            stream_);

            luanchSumBallDummyContactForceTorque(ball_.position(),
            ball_.force(),
            ball_.torque(),
            ballAndDummy_.objectPointed_.neighborPrefixSum(),
            ballAndDummy_.contact_.force(),
            ballAndDummy_.contact_.torque(),
            ballAndDummy_.contact_.point(),
            ball_.deviceSize(),
            ball_.gridDim(),
            ball_.blockDim(),
            stream_);
        }
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
        const std::vector<int> wallMaterial = meshWall_.body_.materialIDHostRef();
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

    pair& getBallPair() { return ballAndBall_.pair_; }

    ball& getBall() { return ball_; }

public:
// ---------------------------------------------------------------------
    // Hertzian: set one material-pair row (append)
    // ---------------------------------------------------------------------
    void setHertzianPair(const int materialIndexA,
                         const int materialIndexB,
                         const double effectiveYoungsModulus,
                         const double effectiveShearModulus,
                         const double restitutionCoefficient,
                         const double rollingStiffnessToShearStiffnessRatio,
                         const double torsionStiffnessToShearStiffnessRatio,
                         const double slidingFrictionCoefficient,
                         const double rollingFrictionCoefficient,
                         const double torsionFrictionCoefficient)
    {
        HertzianRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;

        row.effectiveYoungsModulus = effectiveYoungsModulus;
        row.effectiveShearModulus = effectiveShearModulus;
        row.restitutionCoefficient = restitutionCoefficient;
        row.rollingStiffnessToShearStiffnessRatio = rollingStiffnessToShearStiffnessRatio;
        row.torsionStiffnessToShearStiffnessRatio = torsionStiffnessToShearStiffnessRatio;
        row.slidingFrictionCoefficient = slidingFrictionCoefficient;
        row.rollingFrictionCoefficient = rollingFrictionCoefficient;
        row.torsionFrictionCoefficient = torsionFrictionCoefficient;

        hertzianTable_.push_back(row);
    }

    // ---------------------------------------------------------------------
    // Linear: set one material-pair row (append)
    // ---------------------------------------------------------------------
    void setLinearPair(const int materialIndexA,
                       const int materialIndexB,
                       const double normalStiffness,
                       const double slidingStiffness,
                       const double rollingStiffness,
                       const double torsionStiffness,
                       const double normalDampingCoefficient,
                       const double slidingDampingCoefficient,
                       const double rollingDampingCoefficient,
                       const double torsionDampingCoefficient,
                       const double slidingFrictionCoefficient,
                       const double rollingFrictionCoefficient,
                       const double torsionFrictionCoefficient)
    {
        LinearRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;

        row.normalStiffness = normalStiffness;
        row.slidingStiffness = slidingStiffness;
        row.rollingStiffness = rollingStiffness;
        row.torsionStiffness = torsionStiffness;

        row.normalDampingCoefficient = normalDampingCoefficient;
        row.slidingDampingCoefficient = slidingDampingCoefficient;
        row.rollingDampingCoefficient = rollingDampingCoefficient;
        row.torsionDampingCoefficient = torsionDampingCoefficient;

        row.slidingFrictionCoefficient = slidingFrictionCoefficient;
        row.rollingFrictionCoefficient = rollingFrictionCoefficient;
        row.torsionFrictionCoefficient = torsionFrictionCoefficient;

        linearTable_.push_back(row);
    }

    // ---------------------------------------------------------------------
    // Bonded: set one material-pair row (append)
    // ---------------------------------------------------------------------
    void setBondedPair(const int materialIndexA,
                       const int materialIndexB,
                       const double bondRadiusMultiplier,
                       const double bondYoungsModulus,
                       const double normalToShearStiffnessRatio,
                       const double tensileStrength,
                       const double cohesion,
                       const double frictionCoefficient)
    {
        BondedRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;

        row.bondRadiusMultiplier = bondRadiusMultiplier;
        row.bondYoungsModulus = bondYoungsModulus;
        row.normalToShearStiffnessRatio = normalToShearStiffnessRatio;
        row.tensileStrength = tensileStrength;
        row.cohesion = cohesion;
        row.frictionCoefficient = frictionCoefficient;

        bondedTable_.push_back(row);
    }

    void addBall(const double3 position, 
    const double3 velocity, 
    const double3 angularVelocity, 
    const double radius, 
    const double density, 
    const int materialID, 
    const int clumpID = -1)
    {
        double mass = 4. / 3. * radius * radius * radius * pi() * density;
        double inverseMass = 0.;
        if (mass > 0.0) inverseMass = 1.0 / mass;
        ball_.addHost(position, 
        velocity, 
        angularVelocity, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        radius, 
        inverseMass, 
        materialID, 
        clumpID, 
        -1, 
        -1);
    }

    void addFixedBall(const double3 position, 
    const double radius, 
    const int materialID, 
    const int clumpID = -1)
    {
        ball_.addHost(position, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        radius, 
        0., 
        materialID, 
        clumpID, 
        -1, 
        -1);
    }

    void addClump(std::vector<double3> point, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    double3 velocity, 
    double3 angularVelocity, 
    double mass, 
    symMatrix inertiaTensor, 
    int materialID)
    {
        int clumpID = static_cast<int>(clump_.hostSize());
        size_t pebbleStart = ball_.hostSize();
        size_t pebbleEnd = pebbleStart + point.size();

        std::vector<double3> vel(point.size(), velocity);
        double volume = 0., invMass = 0.;
        if (mass > 1.e-20)
        {
            invMass = 1.0 / mass;
            for (size_t i = 0; i < point.size(); i++)
            {
                volume += 4.0 / 3.0 * pi() * pow(radius[i], 3.0);
                vel[i] += cross(angularVelocity, point[i] - centroidPosition);
            }
        }
        double density_ave = 0;
        if (volume > 0.) density_ave = mass / volume;

        for (size_t i = 0; i < point.size(); i++)
        {
            addBall(point[i], 
            vel[i], 
            angularVelocity, 
            radius[i], 
            density_ave, 
            materialID, 
            clumpID);
        }

        clump_.addHost(centroidPosition, 
        velocity, 
        angularVelocity, 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0,0.0,0.0,0.0), 
        inverse(inertiaTensor), 
        invMass, 
        materialID,
        pebbleStart, 
        pebbleEnd);
    }

    void addFixedClump(std::vector<double3> point, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    int materialID)
    {
        int clumpID = static_cast<int>(clump_.hostSize());
        size_t pebbleStart = ball_.hostSize();
        size_t pebbleEnd = pebbleStart + point.size();

        for (size_t i = 0; i < point.size(); i++)
        {
            addFixedBall(point[i], 
            radius[i], 
            materialID, 
            clumpID);
        }

        clump_.addHost(centroidPosition, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_quaternion(1., 0., 0., 0.), 
        make_symMatrix(0., 0., 0., 0., 0., 0.), 
        0., 
        materialID,
        pebbleStart, 
        pebbleEnd);
    }

    void addMeshWall(const std::vector<double3> &vertexPosition, 
    const std::vector<int> &triIndex0, 
    const std::vector<int> &triIndex1, 
    const std::vector<int> &triIndex2, 
    const double3 &posistion, 
    const double3 &velocity, 
    const double3 &angularVelocity, 
    int matirialID,
    cudaStream_t stream)
    {
        if (triIndex0.size() != triIndex1.size() || triIndex0.size() != triIndex2.size()) return;
        int maxIndex0 = *std::max_element(triIndex0.begin(), triIndex0.end());
        int maxIndex1 = *std::max_element(triIndex1.begin(), triIndex1.end());
        int maxIndex2 = *std::max_element(triIndex2.begin(), triIndex2.end());
        int maxIndex = std::max(maxIndex0, maxIndex1);
        maxIndex = std::max(maxIndex, maxIndex2);
        if (maxIndex >= vertexPosition.size()) return;

        int vertexStart = meshWall_.vertex_.hostSize();
        int wallIndex = meshWall_.body_.hostSize();
        meshWall_.body_.addHost(posistion, 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0, 0.0, 0.0, 0.0), 
        make_symMatrix(0., 0., 0., 0., 0., 0.), 
        0.0, 
        matirialID, 
        vertexStart, 
        vertexStart + vertexPosition.size());

        for (size_t i = 0; i < triIndex0.size(); i++)
        {
            double3 v0 = vertexPosition[triIndex0[i]];
            double3 v1 = vertexPosition[triIndex1[i]];
            double3 v2 = vertexPosition[triIndex2[i]];

            double3 c = triangleCircumcenter(v0, v1, v2);
            meshWall_.triangle_.addHost(triIndex0[i], 
            triIndex1[i], 
            triIndex2[i], 
            wallIndex, 
            c,
            -1, 
            -1);
        }

        for (size_t i = 0; i < vertexPosition.size(); i++)
        {
            meshWall_.vertex_.addHost(vertexPosition[i], 
            vertexPosition[i] + posistion, 
            wallIndex);
        }
    }

    void addBondedPair(const std::vector<int>& objectPointed,
    const std::vector<int>& objectPointing,
    const std::vector<double3>& position)
    {
        bondedBallAndBall_.add(objectPointed, objectPointing, position);
    }

    virtual bool addInitialCondition() 
    {
        return false;
    }

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
        
        if (timeStep <= 0.) 
        {
            std::cout << "failed! Time step is less than 0" << std::endl;
            return;
        }
        size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep;
        if (numFrame > 0) frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        
        initialize(minBoundary, maxBoundary, maximumThread);
        std::cout << "Initialization Completed." << std::endl;
        neighborSearch();
        if (addInitialCondition()) initialize(minBoundary, maxBoundary, maximumThread);

        outputBallVTU(dir, 0, 0, 0.0);
        outputMeshWallVTU(dir, 0, 0, 0.0);

        size_t iStep = 0, iFrame = 0;
        double time = 0.0;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            neighborSearch();
            interaction1stHalf(gravity, timeStep);
            calculateContactForceTorque(timeStep);
            interaction2ndHalf(gravity, timeStep);
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputBallVTU(dir, iFrame, iStep, time);
                outputMeshWallVTU(dir, iFrame, iStep, time);
            }
        }
        ball_.copyDeviceToHost(stream_);
        clump_.copyDeviceToHost(stream_);
        meshWall_.body_.copyDeviceToHost(stream_);
        meshWall_.vertex_.copyDeviceToHost(stream_);
    }

private:
    size_t maxThread_;
    cudaStream_t stream_;

    ball ball_;
    ball dummy_;
    clump clump_;
    meshWall meshWall_;

    std::vector<HertzianRow> hertzianTable_;
    std::vector<LinearRow> linearTable_;
    std::vector<BondedRow> bondedTable_;
    contactModelParameters contactModelParameters_;

    solidInteraction ballAndBall_;
    solidInteraction ballAndDummy_;
    solidInteraction ballAndTriangle_;
    bondedInteraction bondedBallAndBall_;

    spatialGrid ballSpatialGrid_;
    spatialGrid triangleSpatialGrid_;
};