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

extern "C" void updateGridCellStartEnd(spatialGrid& sptialGrids, 
int* hashIndex, 
int* hashValue, 
double3* position, 
const size_t numObjects,
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    sptialGrids.cellInit(stream);

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, numObjects, maxThreadsPerBlock);

    calculateHash <<< grid, block, 0, stream >>> (hashValue, 
    position, 
    sptialGrids.minBound, 
    sptialGrids.maxBound, 
    sptialGrids.cellSize, 
    sptialGrids.gridSize, 
    sptialGrids.deviceSize(), 
    numObjects);
    CUDA_CHECK(cudaGetLastError());

    buildHashStartEnd(sptialGrids.cellHashStart(),
    sptialGrids.cellHashEnd(),
    hashIndex, 
    hashValue, 
    static_cast<int>(sptialGrids.deviceSize()),
    numObjects,
    maxThreadsPerBlock, 
    stream);
}

__global__ void setBallInteractionsKernel(int* objectPointed, 
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
int* neighborCountA, 
int* neighborPrefixSumA, 
int* interactionStartB, 
int* interactionEndB,
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numGrids,
const size_t flag,
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;

    neighborCountA[idxA] = 0;
    if(flag == 0) neighborPrefixSumA[idxA] = 0;

    int count = 0;
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
                if(gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if(gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                int endIndex = cellEnd[hashB];
                if (startIndex == 0xFF) continue;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = ballHashIndex[i];
                    if (idxA >= idxB) continue;
                    if (clumpID[idxA] >= 0 && clumpID[idxA] == clumpID[idxB]) continue;
					if (invMass[idxA] < 1.e-20 && invMass[idxB] < 1.e-20) continue;
                    double3 posB = ballPosition[idxB];
                    double radB = radius[idxB];
                    double3 rAB = posA - posB;
                    double dis = length(rAB);
                    double overlap = 1.1 * (radB + radA) - dis;
                    if (overlap >= 0.)
                    {
                        if (flag == 0) count++;
                        else
                        {
                            int offset_w = atomicAdd(&neighborCountA[idxA], 1);
                            int index_w = base_w + offset_w;
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
                                    if (j1 < 0) return;
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
                        }
                    }
                }
            }
        }
    }
    if (flag == 0) neighborCountA[idxA] = count;
}

extern "C" void launchBallNeighborSearch(solidInteraction& ballInteractions, 
interactionMap &ballInteractionMap,
ball& balls, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    if(spatialGrids.deviceSize() == 0) return;
    updateGridCellStartEnd(spatialGrids,
    balls.hashIndex(),
    balls.hashValue(),
    balls.position(),
    balls.deviceSize(),
    maxThreadsPerBlock,
    stream);

    //debug_dump_device_array(spatialGrids.cellHashStart(), spatialGrids.deviceSize(), "spatialGrids.cellHashStart");
    //debug_dump_device_array(spatialGrids.cellHashEnd(), spatialGrids.deviceSize(), "spatialGrids.cellHashEnd");

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, balls.deviceSize(), maxThreadsPerBlock);

    for (size_t flag = 0; flag < 2; flag++)
    {
        setBallInteractionsKernel <<<grid, block, 0, stream>>> (ballInteractions.objectPointed(),
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
        ballInteractionMap.countA(),
        ballInteractionMap.prefixSumA(),
        ballInteractionMap.startB(),
        ballInteractionMap.endB(),
        spatialGrids.cellHashStart(),
        spatialGrids.cellHashEnd(),
        spatialGrids.minBound,
        spatialGrids.cellSize,
        spatialGrids.gridSize,
        spatialGrids.deviceSize(),
        flag,
        balls.deviceSize());

        if (flag == 0)
        {
            //debug_dump_device_array(ballInteractionMap.countA(), ballInteractionMap.ASize(), "ballInteractionMap.countA");
            int activeNumber = 0;
            auto exec = thrust::cuda::par.on(stream);
            thrust::inclusive_scan(exec,
            thrust::device_pointer_cast(ballInteractionMap.countA()),
            thrust::device_pointer_cast(ballInteractionMap.countA() + ballInteractionMap.ASize()),
            thrust::device_pointer_cast(ballInteractionMap.prefixSumA()));
            cuda_copy_sync(&activeNumber, ballInteractionMap.prefixSumA() + ballInteractionMap.ASize() - 1, 1, CopyDir::D2H);
            ballInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);
        }
    }

    ballInteractionMap.hashInit(ballInteractions.objectPointing(), ballInteractions.activeSize(), stream);
    buildHashStartEnd(ballInteractionMap.startB(), 
    ballInteractionMap.endB(), 
    ballInteractionMap.hashIndex(), 
    ballInteractionMap.hashValue(),
    static_cast<int>(ballInteractionMap.BSize()),
    ballInteractionMap.activeHashSize(),
    maxThreadsPerBlock, 
    stream);
}