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

void sortKeyValuePairs(int* d_keys, int* d_values,
                       std::size_t numObjects,
                          cudaStream_t stream)
{
    auto exec = thrust::cuda::par.on(stream);

    thrust::sort_by_key(exec,d_keys, d_keys + numObjects,
                        d_values);
}

void inclusiveScan(int* prefixSum,
                          int* count,
                          std::size_t num,
                          cudaStream_t stream)
{
    if (num < 1) return;

    auto exec = thrust::cuda::par.on(stream);

    thrust::inclusive_scan(exec,
        thrust::device_pointer_cast(count),
        thrust::device_pointer_cast(count + num),
        thrust::device_pointer_cast(prefixSum));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)       // sm 6.0+
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	return atomicAdd(addr, val);
}
#else                                                   
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	auto  addr_ull = reinterpret_cast<unsigned long long*>(addr);
	unsigned long long old = *addr_ull, assumed;

	do {
		assumed = old;
		double  old_d = __longlong_as_double(assumed);
		double  new_d = old_d + val;
		old = atomicCAS(addr_ull, assumed, __double_as_longlong(new_d));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v)
{
    atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}