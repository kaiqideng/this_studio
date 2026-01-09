#pragma once
#include <cstddef>

inline bool setGPUGridBlockDim(size_t& gridDim, 
size_t& blockDim, 
const size_t numObjects,
const size_t maxThreadsPerBlock)
{
    if (numObjects == 0 || maxThreadsPerBlock == 0) return false;
    blockDim = maxThreadsPerBlock < numObjects ? maxThreadsPerBlock : numObjects;
    gridDim = (numObjects + blockDim - 1) / blockDim;
    return true;
}