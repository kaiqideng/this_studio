#pragma once
#include <vector>
#include <utility>      // std::move, std::exchange
#include <algorithm>    // std::copy
#include "myCUDA.h"

/**
 * @brief 1D array stored on both Host & Device (RAII, move-only).
 * Host:  std::vector<T> h_data
 * Device: T* d_ptr + d_size
 */
template <typename T>
struct HostDeviceArray1D
{
private:
    std::vector<T> h_data;  // host data
    size_t d_size{0};       // number of elements on device

public:
    T* d_ptr{nullptr};      // device pointer

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

            d_ptr  = std::exchange(other.d_ptr,  nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const { return h_data.size(); }
    size_t deviceSize() const{ return d_size; }

    // ---------------------------------------------------------------------
    // Device memory management
    // ---------------------------------------------------------------------
    void releaseDeviceArray()
    {
        if (d_ptr)
        {
            CUDA_FREE(d_ptr);
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    void allocDeviceArray(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) releaseDeviceArray();

        d_size = n;
        CUDA_ALLOC(d_ptr, n, InitMode::ZERO, stream);
    }

    // ---------------------------------------------------------------------
    // Host-side modifications
    // ---------------------------------------------------------------------
    void addHostData(const T& value)
    {
        h_data.push_back(value);
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
        if (newData.size() <= hostSize())
        {
            std::copy(newData.begin(), newData.end(), h_data.begin());
        }
        else
        {
            std::copy(newData.begin(), newData.begin() + hostSize(), h_data.begin());
        }
    }

    // ---------------------------------------------------------------------
    // Host <-> Device transfer
    // ---------------------------------------------------------------------
    void download(cudaStream_t stream)
    {
        const size_t h_size = hostSize();

        if (h_size != d_size)
        {
            allocDeviceArray(h_size, stream);
        }

        if (h_size > 0)
        {
            cuda_copy(d_ptr, h_data.data(), h_size, CopyDir::H2D, stream);
        }
    }

    void upload(cudaStream_t stream)
    {
        if (hostSize() != d_size)
        {
            h_data.resize(d_size);
        }

        if (d_size > 0 && d_ptr)
        {
            cuda_copy(h_data.data(), d_ptr, d_size, CopyDir::D2H, stream);
        }
    }

    const std::vector<T> getHostData()
    {
        if(d_size > 0 && d_ptr)
        {
            if (d_size <= hostSize())
            {
                cuda_copy_sync(h_data.data(), d_ptr, d_size, CopyDir::D2H);
            }
            else 
            {
                cuda_copy_sync(h_data.data(), d_ptr, hostSize(), CopyDir::D2H);
            }
        }

        return h_data;
    }
};

template <typename T>
struct DeviceArray1D
{
private:
    size_t d_size{0};       // number of elements on device

public:
    T* d_ptr{nullptr};      // device pointer

    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    DeviceArray1D() = default;

    ~DeviceArray1D()
    {
        releaseDeviceArray();
    }

    DeviceArray1D(const DeviceArray1D&) = delete;
    DeviceArray1D& operator=(const DeviceArray1D&) = delete;

    DeviceArray1D(DeviceArray1D&& other) noexcept
    {
        *this = std::move(other);
    }

    DeviceArray1D& operator=(DeviceArray1D&& other) noexcept
    {
        if (this != &other)
        {
            releaseDeviceArray();

            d_ptr  = std::exchange(other.d_ptr,  nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const{ return d_size; }

    // ---------------------------------------------------------------------
    // Device memory management
    // ---------------------------------------------------------------------
    void releaseDeviceArray()
    {
        if (d_ptr)
        {
            CUDA_FREE(d_ptr);
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    void allocDeviceArray(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) releaseDeviceArray();

        d_size = n;
        CUDA_ALLOC(d_ptr, n, InitMode::ZERO, stream);
    }
};