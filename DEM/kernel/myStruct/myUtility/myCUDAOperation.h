#pragma once
#include "myCUDA.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

template <typename T>
inline void device_fill(T* d_ptr, size_t n, T value, cudaStream_t stream = 0);

inline void computeGPUGridSizeBlockSize(size_t& gridDim, size_t& blockDim, 
    const size_t numObjects,
    const size_t maxThreadsPerBlock)
{
    if (numObjects == 0)
    {
        gridDim = 1;
        blockDim = 1;
        return;
    }
    blockDim = maxThreadsPerBlock < numObjects ? maxThreadsPerBlock : numObjects;
    gridDim = (numObjects + blockDim - 1) / blockDim;
}

void sortKeyValuePairs(int* d_keys, int* d_values, std::size_t numObjects, cudaStream_t stream);

void inclusiveScan(int* prefixSum, int* count, std::size_t num, cudaStream_t stream);

__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v);