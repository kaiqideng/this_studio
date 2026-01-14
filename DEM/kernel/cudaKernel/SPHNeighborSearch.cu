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
const size_t num)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= num) return;
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
                if (gridPositionB.x < 0 || gridPositionB.y < 0 || gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y || gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = SPHHashIndex[i];
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
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
const size_t num)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= num) return;

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
                if (gridPositionB.x < 0 || gridPositionB.y < 0 || gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y || gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                int countInOneCell = 0;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = SPHHashIndex[i];
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
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
spatialGrid& spatialGrids,
int* SPHHashIndex,
int* SPHHashValue,
double3* SPHPosition,
const double* SPHSmoothLength,
const size_t num,
const size_t maxThreadsPerBlock,
cudaStream_t stream)
{
    size_t gridD = 1, blockD = 1;
    if (setGPUGridBlockDim(gridD, blockD, num, maxThreadsPerBlock))
    {
        updateGridCellStartEnd(spatialGrids,
        SPHHashIndex,
        SPHHashValue,
        SPHPosition,
        num,
        gridD,
        blockD,
        stream);

        countSPHInteractionsKernel <<<gridD, blockD, 0, stream>>> (SPHPosition,
        SPHSmoothLength,
        SPHHashIndex,
        SPHInteractionMap.countA(),
        spatialGrids.cellHashStart(),
        spatialGrids.cellHashEnd(),
        spatialGrids.minBound,
        spatialGrids.cellSize,
        spatialGrids.gridSize,
        num);

        buildPrefixSum(SPHInteractionMap.prefixSumA(), 
        SPHInteractionMap.countA(), 
        SPHInteractionMap.ASize(), 
        stream);
        int activeNumber = 0;
        cuda_copy_sync(&activeNumber, SPHInteractionMap.prefixSumA() + SPHInteractionMap.ASize() - 1, 1, CopyDir::D2H);
        SPHInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);

        writeSPHInteractionsKernel <<<gridD, blockD, 0, stream>>> (SPHInteractions.objectPointed(),
        SPHInteractions.objectPointing(),
        SPHPosition,
        SPHSmoothLength,
        SPHHashIndex,
        SPHInteractionMap.prefixSumA(),
        spatialGrids.cellHashStart(),
        spatialGrids.cellHashEnd(),
        spatialGrids.minBound,
        spatialGrids.cellSize,
        spatialGrids.gridSize,
        num);
    }
}