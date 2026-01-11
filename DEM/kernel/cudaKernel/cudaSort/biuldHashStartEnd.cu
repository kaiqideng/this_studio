#include "buildHashStartEnd.h"

__global__ void setHashIndex(int* hashIndex, 
const size_t hashListSize)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;
    hashIndex[index] = static_cast<int>(index);
}

__global__ void findStartAndEnd(int* start, 
int* end, 
int* hashValue, 
const int maxHashValue,
const size_t hashListSize)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;
    int h = hashValue[index];
    if (h < 0 || h >= maxHashValue) return;
    if (index == 0)
    {
        start[h] = 0;
    }
    else
    {
        int a = hashValue[index - 1];
        if (a >= 0 && a < maxHashValue && a != h)
        {
            start[h] = static_cast<int>(index);
            end[a] = static_cast<int>(index);
        }
    }
    if (index == hashListSize - 1) end[h] = static_cast<int>(hashListSize);
}

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* hashIndex, 
int* hashValue, 
const int maxHashValue, 
const size_t hashListSize, 
const size_t gridDim, 
const size_t blockDim, 
cudaStream_t stream)
{
    cudaMemsetAsync(start, 0xFF, static_cast<size_t>(maxHashValue) * sizeof(int), stream);
    cudaMemsetAsync(end, 0xFF, static_cast<size_t>(maxHashValue) * sizeof(int), stream);

    setHashIndex <<<gridDim, blockDim, 0, stream>>> (hashIndex, hashListSize);

    auto exec = thrust::cuda::par.on(stream);
    try
    {
        cudaError_t err0 = cudaGetLastError();
        if (err0 != cudaSuccess)
        {
            std::cerr << "[buildHashStartEnd] before sort, cudaGetLastError = "
                      << cudaGetErrorString(err0) << "\n";
        }

        thrust::sort_by_key(exec,
        hashValue,
        hashValue + hashListSize,
        hashIndex);

        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
        {
            std::cerr << "[buildHashStartEnd] after sort, cudaGetLastError = "
                      << cudaGetErrorString(err1) << "\n";
        }
    }
    catch (thrust::system_error& e)
    {
        std::cerr << "thrust::sort_by_key threw: " << e.what() << "\n";
        throw;
    }

    findStartAndEnd <<<gridDim, blockDim, 0, stream>>> (start, end, hashValue, maxHashValue, hashListSize);
}

extern "C" void buildPrefixSum(int* prefixSum, 
int* count, 
const size_t size, 
cudaStream_t stream)
{
    auto exec = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(exec,
    thrust::device_pointer_cast(count),
    thrust::device_pointer_cast(count + size),
    thrust::device_pointer_cast(prefixSum));
}