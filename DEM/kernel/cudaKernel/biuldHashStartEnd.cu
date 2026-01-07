#include "buildHashStartEnd.h"

__global__ void setInitialIndices(int* initialIndices, 
const size_t hashListSize)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;
    initialIndices[index] = static_cast<int>(index);
}

__global__ void findStartAndEnd(int* start, 
int* end, 
int* hash, 
const int maxHashValue,
const size_t hashListSize)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;
    int h = hash[index];
    if (h < 0 || h >= maxHashValue) return;
    if (index == 0)
    {
        start[h] = 0;
    }
    else
    {
        int a = hash[index - 1];
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
int* index, 
int* hash, 
const int maxHashValue,
const size_t hashListSize, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    if (hashListSize < 1) return;
    //debug_dump_device_array(hash, hashListSize, "hash");
    CUDA_CHECK(cudaMemsetAsync(start, 0xFF, static_cast<size_t>(maxHashValue) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(end, 0xFF, static_cast<size_t>(maxHashValue) * sizeof(int), stream));

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, hashListSize, maxThreadsPerBlock);
    setInitialIndices <<<grid, block, 0, stream>>> (index, hashListSize);
    CUDA_CHECK(cudaGetLastError());

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
                            hash,
                            hash + hashListSize,
                            index);

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

    findStartAndEnd <<<grid, block, 0, stream>>> (start, end, hash, maxHashValue, hashListSize);
    CUDA_CHECK(cudaGetLastError());
}