#include "buildHashStartEnd.h"

__global__ void setInitialIndices(int* initialIndices, 
const size_t numObjects)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numObjects) return;
    initialIndices[index] = static_cast<int>(index);
}

__global__ void setHashAux(int* aux, 
int* hash, 
const size_t numObjects)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numObjects) return;
    if (index == 0) aux[0] = hash[numObjects - 1];
    if (index > 0)  aux[index] = hash[index - 1];
}

__global__ void findStartAndEnd(int* start, 
int* end, 
int* hash, 
int* aux, 
const size_t numObjects)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numObjects) return;
    if (index == 0 || hash[index] != aux[index])
    {
        start[hash[index]] = static_cast<int>(index);
        end[aux[index]] = static_cast<int>(index);
    }
    if (index == numObjects - 1) end[hash[index]] = static_cast<int>(numObjects);
}

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* index, 
int* hash, 
int* aux, 
const size_t numObjects, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    if (numObjects < 1) return;

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, numObjects, maxThreadsPerBlock);
    
    setInitialIndices <<<grid, block, 0, stream>>> (index, numObjects);
    CUDA_CHECK(cudaGetLastError());

    sortKeyValuePairs(hash, index, numObjects, stream);
    CUDA_CHECK(cudaGetLastError());

    setHashAux <<<grid, block, 0, stream>>> (aux, hash, numObjects);
    CUDA_CHECK(cudaGetLastError());

    findStartAndEnd <<<grid, block, 0, stream>>> (start, end, hash, aux, numObjects);
    CUDA_CHECK(cudaGetLastError());
}