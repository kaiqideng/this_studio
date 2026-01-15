#include "ballNeighborSearch.h"

__global__ void calculateHash(int* hashValue, 
double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numGrids,
const size_t numObjects)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObjects) return;
    double3 p = position[idx];
    if (minBound.x <= p.x && p.x < maxBound.x &&
    minBound.y <= p.y && p.y < maxBound.y &&
    minBound.z <= p.z && p.z < maxBound.z)
    {
        int3 gridPosition = calculateGridPosition(p, minBound, cellSize);
        hashValue[idx] = calculateHash(gridPosition, gridSize);
    }
    else
    {
        hashValue[idx] = numGrids - 1;
    }
}

extern "C" void updateGridCellStartEnd(spatialGrid& spatialGrids, 
int* hashIndex, 
int* hashValue, 
double3* position, 
const size_t numObjects,
const size_t gridD, 
const size_t blockD, 
cudaStream_t stream)
{
    calculateHash <<< gridD, blockD, 0, stream >>> (hashValue, 
    position, 
    spatialGrids.minBound, 
    spatialGrids.maxBound, 
    spatialGrids.cellSize, 
    spatialGrids.gridSize, 
    spatialGrids.deviceSize(), 
    numObjects);
    CUDA_CHECK(cudaGetLastError());

    //debug_dump_device_array(hashValue, numObjects, "hashValue");
    CUDA_CHECK(cudaMemsetAsync(spatialGrids.cellHashStart(), 0xFF, spatialGrids.deviceSize() * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(spatialGrids.cellHashEnd(), 0xFF, spatialGrids.deviceSize() * sizeof(int), stream));
    buildHashStartEnd(spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    hashIndex,
    hashValue,
    spatialGrids.deviceSize(),
    numObjects,
    gridD,
    blockD, 
    stream);
}

__global__ void countBallInteractionsKernel(int* neighborCountA, 
double3* ballPosition, 
const double* radius, 
const double* invMass, 
const int* clumpID, 
int* ballHashIndex, 
int* cellHashStart,
int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;
    int count = 0;

    double3 posA = ballPosition[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 || gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y || gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = ballHashIndex[i];
                    if (idxA >= idxB) continue;
                    if (clumpID[idxA] >= 0 && clumpID[idxA] == clumpID[idxB]) continue;
					if (invMass[idxA] < 1.e-20 && invMass[idxB] < 1.e-20) continue;
                    double cut = 1.1 * (radA + radius[idxB]);
                    double3 posB = ballPosition[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCountA[idxA] = count;
}

__global__ void writeBallInteractionsKernel(int* objectPointed, 
int* objectPointing, 
double3* contactForce, 
double3* contactTorque, 
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
int* objectPointed_history, 
double3* slidingSpring_history, 
double3* rollingSpring_history, 
double3* torsionSpring_history, 
double3* ballPosition, 
const double* radius, 
const double* invMass, 
const int* clumpID, 
int* ballHashIndex, 
int* interactionMapHashIndex,
int* neighborPrefixSumA, 
int* interactionStartB, 
int* interactionEndB,
int* cellHashStart, 
int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;

    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSumA[idxA - 1];
    double3 posA = ballPosition[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 || gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y || gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellHashEnd[hashB];
                int countInOneCell = 0;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = ballHashIndex[i];
                    if (idxA >= idxB) continue;
                    if (clumpID[idxA] >= 0 && clumpID[idxA] == clumpID[idxB]) continue;
					if (invMass[idxA] < 1.e-20 && invMass[idxB] < 1.e-20) continue;
                    double cut = 1.1 * (radA + radius[idxB]);
                    double3 posB = ballPosition[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + countInOneCell;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        contactForce[index_w] = make_double3(0, 0, 0);
                        contactTorque[index_w] = make_double3(0, 0, 0);
                        slidingSpring[index_w] = make_double3(0, 0, 0);
                        rollingSpring[index_w] = make_double3(0, 0, 0);
                        torsionSpring[index_w] = make_double3(0, 0, 0);
                        if (interactionStartB[idxB] != 0xFFFFFFFF)
                        {
                            for (int j = interactionStartB[idxB]; j < interactionEndB[idxB]; j++)
                            {
                                int j1 = interactionMapHashIndex[j];
                                int idxA1 = objectPointed_history[j1];
                                if (idxA == idxA1)
                                {
                                    slidingSpring[index_w] = slidingSpring_history[j1];
                                    rollingSpring[index_w] = rollingSpring_history[j1];
                                    torsionSpring[index_w] = torsionSpring_history[j1];
                                    break;
                                }
                            }
                        }
                        countInOneCell++;
                    }
                }
            }
        }
    }
}

extern "C" void launchBallNeighborSearch(solidInteraction& ballInteractions, 
interactionMap &ballInteractionMap,
ball& balls, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock,
cudaStream_t stream)
{
    int activeNumber = 0;
    size_t gridD = 1, blockD = 1;
    if (setGPUGridBlockDim(gridD, blockD, balls.deviceSize(), maxThreadsPerBlock))
    {
        updateGridCellStartEnd(spatialGrids,
        balls.hashIndex(),
        balls.hashValue(),
        balls.position(),
        balls.deviceSize(),
        gridD, 
        blockD, 
        stream);

        ballInteractions.updateHistory(stream);

        countBallInteractionsKernel <<<gridD, blockD, 0, stream>>> (ballInteractionMap.countA(),
        balls.position(),
        balls.radius(),
        balls.inverseMass(),
        balls.clumpID(),
        balls.hashIndex(),
        spatialGrids.cellHashStart(),
        spatialGrids.cellHashEnd(),
        spatialGrids.minBound,
        spatialGrids.cellSize,
        spatialGrids.gridSize,
        balls.deviceSize());

        //debug_dump_device_array(ballInteractionMap.countA(), ballInteractionMap.ASize(), "ballInteractionMap.countA");
        buildPrefixSum(ballInteractionMap.prefixSumA(), 
        ballInteractionMap.countA(), 
        ballInteractionMap.ASize(), 
        stream);
        cuda_copy_sync(&activeNumber, ballInteractionMap.prefixSumA() + ballInteractionMap.ASize() - 1, 1, CopyDir::D2H);
        ballInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);

        writeBallInteractionsKernel <<<gridD, blockD, 0, stream>>> (ballInteractions.objectPointed(),
        ballInteractions.objectPointing(),
        ballInteractions.force(),
        ballInteractions.torque(),
        ballInteractions.slidingSpring(),
        ballInteractions.rollingSpring(),
        ballInteractions.torsionSpring(),
        ballInteractions.objectPointedHistory(),
        ballInteractions.slidingSpringHistory(),
        ballInteractions.rollingSpringHistory(),
        ballInteractions.torsionSpringHistory(),
        balls.position(),
        balls.radius(),
        balls.inverseMass(),
        balls.clumpID(),
        balls.hashIndex(),
        ballInteractionMap.hashIndex(),
        ballInteractionMap.prefixSumA(),
        ballInteractionMap.startB(),
        ballInteractionMap.endB(),
        spatialGrids.cellHashStart(),
        spatialGrids.cellHashEnd(),
        spatialGrids.minBound,
        spatialGrids.cellSize,
        spatialGrids.gridSize,
        balls.deviceSize());
    }

    CUDA_CHECK(cudaMemsetAsync(ballInteractionMap.startB(), 0xFF, ballInteractionMap.BSize() * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(ballInteractionMap.endB(), 0xFF, ballInteractionMap.BSize() * sizeof(int), stream));
    ballInteractionMap.resizeHashIndex(static_cast<size_t>(activeNumber), stream);
    if (setGPUGridBlockDim(gridD, blockD, static_cast<size_t>(activeNumber), maxThreadsPerBlock))
    {
        buildHashStartEnd(ballInteractionMap.startB(), 
        ballInteractionMap.endB(), 
        ballInteractionMap.hashIndex(), 
        ballInteractions.objectPointing(),
        ballInteractionMap.BSize(),
        static_cast<size_t>(activeNumber),
        gridD,
        blockD,
        stream);
    }
}