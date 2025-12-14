#pragma once
#include "myStruct/myUtility/myCUDAOperation.h"

__device__ __forceinline__ int3 calculateGridPosition(double3 position, const double3 minBoundary, const double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
        int((position.y - minBoundary.y) / cellSize.y),
        int((position.z - minBoundary.z) / cellSize.z));
}

__device__ __forceinline__ int calculateHash(int3 gridPosition, const int3 gridSize)
{
    if(gridPosition.x < 0 || gridPosition.y < 0 || gridPosition.z < 0 ) return -1;
    return gridPosition.z * gridSize.y * gridSize.x + gridPosition.y * gridSize.x + gridPosition.x;
}

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* index, 
int* hash, 
int* aux, 
const size_t numObjects, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);