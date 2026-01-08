#include "SPHNeighborSearch.h"

__global__ void countSPHInteractionsKernel(double3* position, 
const double* smoothLength, 
int* SPHHashIndex, 
int* neighborCountA, 
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPHs,
const size_t numGhosts)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPHs + numGhosts) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = SPHHashIndex[i];
                    if (idxA == idxB) continue;
                    if (idxA >= numSPHs && idxB >= numSPHs) continue;
                    double cut = 2.0 * radA;
                    double3 posB = position[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCountA[idxA] = count;
}

__global__ void writeSPHInteractionsKernel(int* objectPointed, 
int* objectPointing, 
double3* position, 
const double* smoothLength, 
int* SPHHashIndex, 
int* neighborPrefixSumA, 
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPHs,
const size_t numGhosts)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPHs + numGhosts) return;

    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSumA[idxA - 1];
    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                int countInOneCell = 0;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = SPHHashIndex[i];
                    if (idxA == idxB) continue;
                    if (idxA >= numSPHs && idxB >= numSPHs) continue;
                    double cut = 2.0 * radA;
                    double3 posB = position[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + countInOneCell;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        countInOneCell++;
                    }
                }
            }
        }
    }
}

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
SPH& SPHAndGhosts, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    size_t numSPHs = SPHAndGhosts.SPHDeviceSize();
    size_t numGhosts = SPHAndGhosts.ghostDeviceSize();

    size_t grid = 1, block = 1;

    updateGridCellStartEnd(spatialGrids,
    SPHAndGhosts.hashIndex(),
    SPHAndGhosts.hashValue(),
    SPHAndGhosts.position(),
    numSPHs + numGhosts,
    maxThreadsPerBlock,
    stream);

    computeGPUGridSizeBlockSize(grid, block, numSPHs + numGhosts, maxThreadsPerBlock);
    countSPHInteractionsKernel <<<grid, block, 0, stream>>> (SPHAndGhosts.position(),
    SPHAndGhosts.smoothLength(),
    SPHAndGhosts.hashIndex(),
    SPHInteractionMap.countA(),
    spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    spatialGrids.minBound,
    spatialGrids.cellSize,
    spatialGrids.gridSize,
    numSPHs,
    numGhosts);

    int activeNumber = 0;
    auto exec = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(exec,
    thrust::device_pointer_cast(SPHInteractionMap.countA()),
    thrust::device_pointer_cast(SPHInteractionMap.countA() + SPHInteractionMap.ASize()),
    thrust::device_pointer_cast(SPHInteractionMap.prefixSumA()));
    cuda_copy_sync(&activeNumber, SPHInteractionMap.prefixSumA() + SPHInteractionMap.ASize() - 1, 1, CopyDir::D2H);
    SPHInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);

    writeSPHInteractionsKernel <<<grid, block, 0, stream>>> (SPHInteractions.objectPointed(),
    SPHInteractions.objectPointing(),
    SPHAndGhosts.position(),
    SPHAndGhosts.smoothLength(),
    SPHAndGhosts.hashIndex(),
    SPHInteractionMap.prefixSumA(),
    spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    spatialGrids.minBound,
    spatialGrids.cellSize,
    spatialGrids.gridSize,
    numSPHs,
    numGhosts);
}