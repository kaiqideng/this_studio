#include "myCUDAOperation.h"

template <typename T>
__global__ void device_fill_kernel(T* data, size_t n, T value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

template <typename T>
inline void device_fill(T* d_ptr, size_t n, T value, cudaStream_t stream)
{
    if (!d_ptr || n == 0) return;

    const int block = 256;
    const int grid  = static_cast<int>((n + block - 1) / block);

    device_fill_kernel<<<grid, block, 0, stream>>>(d_ptr, n, value);
    CUDA_CHECK(cudaGetLastError());
}