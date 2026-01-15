#pragma once
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "myMat.h"

inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line)
{
    if (result != cudaSuccess) 
    {
        std::string msg = std::string("CUDA Error at ") + file + ":" + std::to_string(line) +
        " / " + func + ": " + cudaGetErrorString(result);
        // Throw exception to allow destructors (RAII) to clean up
        throw std::runtime_error(msg);
    }
}

#define CUDA_CHECK(val) check_cuda_error((val), #val, __FILE__, __LINE__)

template <typename T>
struct HostDeviceArray1D
{
private:
    std::vector<T> h_data; // host data
    size_t d_size {0}; // number of elements on device

public:
    T* d_ptr {nullptr}; // device pointer

    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    HostDeviceArray1D() = default;

    ~HostDeviceArray1D()
    {
        releaseDeviceArray();
    }

    HostDeviceArray1D(const HostDeviceArray1D&) = delete;
    HostDeviceArray1D& operator=(const HostDeviceArray1D&) = delete;

    HostDeviceArray1D(HostDeviceArray1D&& other) noexcept
    {
        *this = std::move(other);
    }

    HostDeviceArray1D& operator=(HostDeviceArray1D&& other) noexcept
    {
        if (this != &other)
        {
            releaseDeviceArray();

            h_data = std::move(other.h_data);

            d_ptr = std::exchange(other.d_ptr,  nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const { return h_data.size(); }
    size_t deviceSize() const { return d_size; }

    // ---------------------------------------------------------------------
    // Device memory management
    // ---------------------------------------------------------------------
    void releaseDeviceArray()
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFreeAsync(d_ptr));
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    void allocDeviceArray(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) releaseDeviceArray();

        d_size = n;
        CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, d_size * sizeof(T), stream));
    }

    // ---------------------------------------------------------------------
    // Host-side modifications
    // ---------------------------------------------------------------------
    void addHostData(const T& value)
    {
        h_data.push_back(value);
    }

    void insertHostData(size_t index, const T& value)
    {
        if (index > hostSize()) { h_data.push_back(value); return; }
        h_data.insert(h_data.begin() + static_cast<std::ptrdiff_t>(index), value);
    }

    void removeHostData(size_t index)
    {
        if (index >= h_data.size()) return;
        h_data.erase(h_data.begin() + index);
    }

    void clearHostData()
    {
        h_data.clear();
    }

    void setHostData(const std::vector<T>& newData)
    {
        h_data = newData;
    }

    // ---------------------------------------------------------------------
    // Host <-> Device transfer
    // ---------------------------------------------------------------------
    
    // host -> device
    void upload(cudaStream_t stream) 
    {
        const size_t h_size = hostSize();

        if (h_size != d_size)
        {
            allocDeviceArray(h_size, stream);
        }
        if (h_size > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_data.data(), h_size * sizeof(T), cudaMemcpyHostToDevice, stream));
        }
    }

    // device -> host
    void download(cudaStream_t stream)
    {
        if (d_size > hostSize())
        {
            h_data.resize(d_size);
        }
        if (d_size > 0 && d_ptr)
        {
            CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_ptr, d_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }
    }

    // device -> host
    std::vector<T> getHostData()
    {
        if (d_size > 0 && d_ptr)
        {
            if (hostSize() < d_size)
            {
                h_data.resize(d_size);
            }
            CUDA_CHECK(cudaMemcpy(h_data.data(), d_ptr, d_size * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return h_data;
    }
};