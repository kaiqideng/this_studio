#include "buildHashStartEnd.h"

// Simple CUDA error checker.
static inline void CUDA_CHECK(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) {
        std::printf("[CUDA] %s: %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

// Sort key/value pairs by key (ascending).
// - d_hashValue: int[N] on device (input/output keys)
// - d_hashIndex: int[N] on device (input/output values)
// - N: number of pairs
// - stream: CUDA stream to run the sort on
extern "C"
void sortHashPairsByValue(int* d_hashValue,
int* d_hashIndex,
size_t N,
cudaStream_t stream)
{
    // Basic argument validation.
    if (!d_hashValue || !d_hashIndex) {
        std::printf("sortHashPairsByValue: null device pointer(s)\n");
        std::abort();
    }
    if (N == 0) return;

    // CUB's DeviceRadixSort APIs typically take the item count as 'int'.
    // Guard against extremely large N.
    if (N > static_cast<size_t>(INT_MAX)) {
        std::printf("sortHashPairsByValue: N too large (%zu > INT_MAX)\n", N);
        std::abort();
    }

    // Allocate alternate buffers required by DoubleBuffer for out-of-place radix sort.
    int* d_keys_alt = nullptr;
    int* d_vals_alt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys_alt, N * sizeof(int)), "cudaMalloc d_keys_alt");
    CUDA_CHECK(cudaMalloc(&d_vals_alt, N * sizeof(int)), "cudaMalloc d_vals_alt");

    // Wrap input pointers into CUB DoubleBuffer.
    // CUB will write the result either to the original buffers or to the alternate buffers,
    // and DoubleBuffer::Current() tells us where the final result is.
    cub::DoubleBuffer<int> d_keys(d_hashValue, d_keys_alt);
    cub::DoubleBuffer<int> d_vals(d_hashIndex, d_vals_alt);

    // Query temp storage requirements.
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            d_keys, d_vals,
            static_cast<int>(N),
            /*begin_bit=*/0,
            /*end_bit=*/static_cast<int>(sizeof(int) * 8),
            stream),
        "cub::DeviceRadixSort::SortPairs temp_bytes query"
    );

    // Allocate temp storage and run the sort.
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes), "cudaMalloc d_temp");

    CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            d_keys, d_vals,
            static_cast<int>(N),
            /*begin_bit=*/0,
            /*end_bit=*/static_cast<int>(sizeof(int) * 8),
            stream),
        "cub::DeviceRadixSort::SortPairs exec"
    );

    // Check for any asynchronous kernel launch errors.
    CUDA_CHECK(cudaGetLastError(), "post SortPairs cudaGetLastError");

    // Ensure the output ends up in the user-provided arrays (in/out behavior).
    // If CUB wrote into the alternate buffers, copy results back to the original buffers.
    if (d_keys.Current() != d_hashValue) {
        CUDA_CHECK(cudaMemcpyAsync(d_hashValue, d_keys.Current(),
                                   N * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   stream),
                   "copy back sorted keys");
    }
    if (d_vals.Current() != d_hashIndex) {
        CUDA_CHECK(cudaMemcpyAsync(d_hashIndex, d_vals.Current(),
                                   N * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   stream),
                   "copy back sorted values");
    }

    // Free temporary buffers.
    CUDA_CHECK(cudaFree(d_temp),     "cudaFree d_temp");
    CUDA_CHECK(cudaFree(d_keys_alt), "cudaFree d_keys_alt");
    CUDA_CHECK(cudaFree(d_vals_alt), "cudaFree d_vals_alt");
}

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
const int startEndSize,
const size_t hashListSize)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hashListSize) return;

    int h = hashValue[index];

    if (h < 0 || h >= startEndSize) return;
    if (index == 0 || hashValue[index - 1] != h)
    {
        start[h] = static_cast<int>(index);
    }
    if (index == hashListSize - 1 || hashValue[index + 1] != h) 
    {
        end[h] = static_cast<int>(index + 1);
    }
}

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* hashIndex, 
int* hashValue, 
const int startEndSize, 
const size_t hashListSize, 
const size_t gridD, 
const size_t blockD, 
cudaStream_t stream)
{
    if (gridD * blockD < hashListSize) return;

    setHashIndex <<<gridD, blockD, 0, stream>>> (hashIndex, 
    hashListSize);

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

    /*
    sortHashPairsByValue(hashValue, 
    hashIndex, 
    hashListSize, 
    stream);
     */

    findStartAndEnd <<<gridD, blockD, 0, stream>>> (start, 
    end, 
    hashValue, 
    startEndSize, 
    hashListSize);
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