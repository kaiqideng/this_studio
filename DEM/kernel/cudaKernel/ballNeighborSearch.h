#pragma once
#include "cudaSort/buildHashStartEnd.h"
#include "myStruct/interaction.h"
#include "myStruct/particle.h"
#include "myStruct/spatialGrid.h"
#include "GPUConfig.h"

__device__ __forceinline__ int3 calculateGridPosition(double3 position, const double3 minBoundary, const double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
    int((position.y - minBoundary.y) / cellSize.y),
    int((position.z - minBoundary.z) / cellSize.z));
}

__device__ __forceinline__ int calculateHash(int3 gridPosition, const int3 gridSize)
{
    return gridPosition.z * gridSize.y * gridSize.x + gridPosition.y * gridSize.x + gridPosition.x;
}

extern "C" void updateGridCellStartEnd(spatialGrid& sptialGrids, 
int* hashIndex, 
int* hashValue, 
double3* position, 
const size_t numObjects,
const size_t gridDim, 
const size_t blockDim, 
cudaStream_t stream);

extern "C" void launchBallNeighborSearch(solidInteraction& ballInteractions, 
interactionMap &ballInteractionMap,
ball& balls, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock,
cudaStream_t stream);