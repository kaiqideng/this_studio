#pragma once
#include <cuda_runtime.h>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// Error Handling
// -----------------------------------------------------------------------------

inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::string msg = std::string("CUDA Error at ") + file + ":" + std::to_string(line) +
                          " / " + func + ": " + cudaGetErrorString(result);
        // Throw exception to allow destructors (RAII) to clean up
        throw std::runtime_error(msg);
    }
}

#define CUDA_CHECK(val) check_cuda_error((val), #val, __FILE__, __LINE__)

// -----------------------------------------------------------------------------
// Enums & Constants
// -----------------------------------------------------------------------------

enum class InitMode { 
    NONE,       // Do not initialize
    ZERO,       // Set to 0
    NEG_ONE     // Set to -1 (0xFFFFFFFF for int, NaN for float)
};

enum class CopyDir { 
    H2D,    // Host to Device
    D2H,    // Device to Host
    D2D,    // Device to Device
    AUTO    // Infer from pointer attributes (Unified Virtual Addressing)
};

// -----------------------------------------------------------------------------
// Memory Allocation (Async / Stream Ordered)
// -----------------------------------------------------------------------------

/**
 * @brief Allocates memory on the device using the Stream Ordered Allocator.
 * * @tparam T Type of the element.
 * @param n Number of elements.
 * @param mode Initialization mode (NONE, ZERO, NEG_ONE).
 * @param stream The CUDA stream to associate the allocation with (default 0).
 * @return T* Pointer to the allocated device memory.
 */
template<typename T>
T* cuda_alloc_async(std::size_t n, InitMode mode = InitMode::NONE, cudaStream_t stream = 0) {
    if (n == 0) return nullptr;

    T* ptr = nullptr;

    // Use cudaMallocAsync (Requires CUDA 11.2+)
    // This is faster and prevents CPU-GPU synchronization blocks.
    CUDA_CHECK(cudaMallocAsync(&ptr, n * sizeof(T), stream));

    switch (mode) {
        case InitMode::ZERO:
            CUDA_CHECK(cudaMemsetAsync(ptr, 0, n * sizeof(T), stream));
            break;

        case InitMode::NEG_ONE:
            // 0xFF sets int to -1. 
            // Warning: For float, this creates a NaN.
            CUDA_CHECK(cudaMemsetAsync(ptr, 0xFF, n * sizeof(T), stream));
            break;

        case InitMode::NONE:
        default:
            break;
    }

    return ptr;
}

/**
 * @brief Frees memory using the Stream Ordered Allocator.
 * * @tparam T Type of the element.
 * @param p Reference to the pointer (will be set to nullptr).
 * @param stream The CUDA stream to order the deallocation (default 0).
 */
template<typename T>
void cuda_free_async(T*& p, cudaStream_t stream = 0) {
    if (p) {
        // Asynchronously free memory. The memory is returned to the pool
        // only after previous operations on 'stream' are complete.
        // Note: We use check_cuda_error directly to avoid macro issues in headers sometimes.
        check_cuda_error(cudaFreeAsync(p, stream), "cudaFreeAsync", __FILE__, __LINE__);
        p = nullptr;
    }
}

// -----------------------------------------------------------------------------
// Data Transfer
// -----------------------------------------------------------------------------

/**
 * @brief Async memory copy wrapper.
 */
template<typename T>
inline void cuda_copy(T* dst, const T* src, std::size_t n, CopyDir dir, cudaStream_t stream = 0) {
    if (n == 0) return;

    cudaMemcpyKind kind;
    switch (dir) {
        case CopyDir::H2D:  kind = cudaMemcpyHostToDevice; break;
        case CopyDir::D2H:  kind = cudaMemcpyDeviceToHost; break;
        case CopyDir::D2D:  kind = cudaMemcpyDeviceToDevice; break;
        default:            kind = cudaMemcpyDefault; break;
    }

    CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(T), kind, stream));
}

// -----------------------------------------------------------------------------
// Macros (Updated for 4 arguments)
// -----------------------------------------------------------------------------

// Defines the macro to accept 4 arguments: PTR, N, MODE, STREAM.
// Example: CUDA_ALLOC(myPtr, 100, InitMode::ZERO, myStream);
#define CUDA_ALLOC(PTR, N, MODE, STREAM) \
    (PTR) = cuda_alloc_async<std::remove_pointer_t<decltype(PTR)>>(N, MODE, STREAM)

// Defines the macro for freeing. 
// Uses default stream (0) because destructors usually don't know the stream context.
// If strict ordering is needed, call cuda_free_async directly.
#define CUDA_FREE(PTR) cuda_free_async(PTR, 0)

template <typename T>
void debug_dump_device_array(const T* d_ptr,
                             std::size_t n,
                             const char* name,
                             cudaStream_t stream = 0)
{
    if (n == 0) return;

    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<T> h_buf(n);

    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_ptr,
                          n * sizeof(T),
                          cudaMemcpyDeviceToHost));

    std::cout << "[DEBUG] " << name << " (first " << n << " values):\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "  [" << i << "] = " << h_buf[i] << "\n";
    }
}

template <>
inline void debug_dump_device_array<double3>(const double3* d_ptr, std::size_t n,
                                             const char* name,
                                             cudaStream_t stream)
{
    std::vector<double3> h(n);
    cudaMemcpyAsync(h.data(), d_ptr, n * sizeof(double3), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << name << ":\n";

    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);

    // 科学计数法 + 指定有效数字位数
    std::cout << std::scientific << std::setprecision(3);  // 比如 1.000e-10

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& v = h[i];
        std::cout << "  [" << i << "] = ("
                  << v.x << ", "
                  << v.y << ", "
                  << v.z << ")\n";
    }

    std::cout.copyfmt(old_state);
}

template<typename T>
inline void cuda_copy_sync(T* dst,
                           const T* src,
                           std::size_t n,
                           CopyDir dir)
{
    if (n == 0) return;

    cudaMemcpyKind kind;
    switch (dir) {
        case CopyDir::H2D:  kind = cudaMemcpyHostToDevice; break;
        case CopyDir::D2H:  kind = cudaMemcpyDeviceToHost; break;
        case CopyDir::D2D:  kind = cudaMemcpyDeviceToDevice; break;
        default:            kind = cudaMemcpyDefault;      break;
    }

    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), kind));
}

template <typename T>
inline void device_fill(T* d_ptr, size_t n, T value, cudaStream_t stream = 0);

__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v);