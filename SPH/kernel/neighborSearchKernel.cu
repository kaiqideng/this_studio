#include "neighborSearchKernel.h"

__global__ void calculateHash(int* hashValue, 
const double3* position, 
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

extern "C" void updateSpatialGridCellHashStartEnd(double3* position, 
int* hashIndex, 
int* hashValue, 

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 maxBound,
const double3 cellSize,
const int3 gridSize,
const size_t numGrids,

const size_t numObjects,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    if (gridD_GPU * blockD_GPU < numObjects) return;

    calculateHash <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
    position, 
    minBound, 
    maxBound, 
    cellSize, 
    gridSize, 
    numGrids, 
    numObjects);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(cellHashStart, 0xFF, numGrids * sizeof(int), stream_GPU));
    CUDA_CHECK(cudaMemsetAsync(cellHashEnd, 0xFF, numGrids * sizeof(int), stream_GPU));
    buildHashStartEnd(cellHashStart,
    cellHashEnd,
    hashIndex,
    hashValue,
    numGrids,
    numObjects,
    stream_GPU);
}

__global__ void countSPHInteractionsKernel(int* neighborCount,
const double3* position, 
const double* smoothLength, 
const int* hashIndex,
const int* cellHashStart, 
const int* cellHashEnd,
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
                int startIndex = cellHashStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex[i];
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeSPHInteractionsKernel(int* objectPointed, 
int* objectPointing,
const double3* position, 
const double* smoothLength, 
const int* hashIndex,
const int* neighborPrefixSum, 
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t num)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= num) return;

    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
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
                int startIndex = cellHashStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellHashEnd[hashB];
                int countInOneCell = 0;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex[i];
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

extern "C" void launchSPHNeighborSearch(double3* position, 
double* smoothLength,
int* hashIndex, 
int* hashValue,
int* neighborCount,
int* neighborPrefixSum,

int* cellHashStart,
int* cellHashEnd,

int* objectPointed, 
int* objectPointing,
const size_t maximumPairs,

const double3 minBound,
const double3 maxBound,
const double3 cellSize,
const int3 gridSize,
const size_t numGrids,

const size_t num,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    updateSpatialGridCellHashStartEnd(position,
    hashIndex, 
    hashValue, 
    cellHashStart,
    cellHashEnd,
    minBound,
    maxBound,
    cellSize,
    gridSize,
    numGrids,
    num,
    gridD_GPU, 
    blockD_GPU, 
    stream_GPU);

    countSPHInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    smoothLength,
    hashIndex,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    num);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    num,
    stream_GPU);

    int activeSize = 0;
    cudaMemcpy(&activeSize, neighborPrefixSum + num - 1, sizeof(int), cudaMemcpyDeviceToHost);
    if (activeSize > maximumPairs) 
    {
        if(objectPointed) cudaFree(objectPointed);
        objectPointed = nullptr;
        if(objectPointing) cudaFree(objectPointing);
        objectPointing = nullptr;
        cudaMemsetAsync(objectPointed, 0xff, static_cast<size_t>(activeSize) * sizeof(int), stream_GPU);
        cudaMemsetAsync(objectPointing, 0xff, static_cast<size_t>(activeSize) * sizeof(int), stream_GPU);
    }

    writeSPHInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (objectPointed,
    objectPointing,
    position,
    smoothLength,
    hashIndex,
    neighborPrefixSum,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    num);
}