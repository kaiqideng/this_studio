#pragma once
#include <cstddef>

inline bool setGPUGridBlockDim(size_t& gridD, 
size_t& blockD, 
const size_t numObjects,
const size_t maxThreadsPerBlock)
{
    if (numObjects == 0 || maxThreadsPerBlock == 0) return false;
    blockD = maxThreadsPerBlock < numObjects ? maxThreadsPerBlock : numObjects;
    gridD = (numObjects + blockD - 1) / blockD;
    return true;
}