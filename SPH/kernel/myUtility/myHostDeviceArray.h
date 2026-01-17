#pragma once
#include <cassert>
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
    std::vector<T> h_data;     // host data
    size_t d_size {0};         // number of elements on device

public:
    T* d_ptr {nullptr};        // device pointer

    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    HostDeviceArray1D() = default;

    ~HostDeviceArray1D()
    {
        // Destructor cannot rely on a user-provided stream; use sync free.
        releaseDeviceSync();
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
            // We cannot safely cudaFreeAsync without a stream here; use sync free.
            releaseDeviceSync();

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
    size_t deviceSize() const { return d_size; }

    // ---------------------------------------------------------------------
    // Device memory management
    // ---------------------------------------------------------------------

    // Async free (preferred when you have a stream).
    void releaseDevice(cudaStream_t stream)
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    // Sync free (safe for destructor / move assignment).
    void releaseDeviceSync()
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    // Allocate device array of size n.
    // - Frees old device memory first.
    // - Uses cudaMallocAsync/cudaFreeAsync on the provided stream.
    // - If n==0, releases device memory and returns.
    void allocateDevice(size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        // Release old allocation if any.
        if (d_ptr) releaseDevice(stream);

        d_size = n;
        if (d_size == 0)
        {
            d_ptr = nullptr;
            return;
        }

        CUDA_CHECK(cudaMallocAsync((void**)&d_ptr, d_size * sizeof(T), stream));
        if (zeroFill)
        {
            CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, d_size * sizeof(T), stream));
        }
    }

    // ---------------------------------------------------------------------
    // Host-side modifications
    // ---------------------------------------------------------------------
    void pushHost(const T& value)
    {
        h_data.push_back(value);
    }

    void insertHost(size_t index, const T& value)
    {
        if (index >= hostSize()) { h_data.push_back(value); return; }
        h_data.insert(h_data.begin() + static_cast<std::ptrdiff_t>(index), value);
    }

    void eraseHost(size_t index)
    {
        if (index >= h_data.size()) return;
        h_data.erase(h_data.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void clearHost()
    {
        h_data.clear();
    }

    void reserveHost(size_t n)
    {
        h_data.reserve(n);
    }

    void resizeHost(size_t n)
    {
        h_data.resize(n);
    }

    const std::vector<T>& hostRef() const { return h_data; }

    // ---------------------------------------------------------------------
    // Host <-> Device transfer
    // ---------------------------------------------------------------------

    // Host -> Device (async).
    // Allocates device memory to match hostSize().
    void copyHostToDevice(cudaStream_t stream)
    {
        const size_t n = hostSize();

        if (n != d_size || d_ptr == nullptr)
        {
            // No need to zero-fill because memcpy overwrites the buffer.
            allocateDevice(n, stream, /*zeroFill=*/false);
        }

        if (n > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr,
                                       h_data.data(),
                                       n * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
        }
    }

    // Device -> Host (async).
    // After calling this, synchronize the stream before reading h_data.
    void copyDeviceToHost(cudaStream_t stream)
    {
        if (d_size == 0 || d_ptr == nullptr) return;

        if (d_size != hostSize())
        {
            h_data.resize(d_size);
        }

        CUDA_CHECK(cudaMemcpyAsync(h_data.data(),
                                   d_ptr,
                                   d_size * sizeof(T),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    }

    // Device -> Host (sync).
    std::vector<T> getHostCopy()
    {
        if (d_size > 0 && d_ptr)
        {
            if (d_size != hostSize())
            {
                h_data.resize(d_size);
            }
            CUDA_CHECK(cudaMemcpy(h_data.data(),
                                  d_ptr,
                                  d_size * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return h_data;
    }

    void setHost(const std::vector<T>& newData)
    {
        h_data = newData;
    }
};

template <typename... Ts>
struct SoA1D
{
private:
    std::tuple<HostDeviceArray1D<Ts>...> fields_;

private:
    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------
    static constexpr size_t numFields_ {sizeof...(Ts)};

    template <typename F, size_t... Is>
    void forEach_(F&& f, std::index_sequence<Is...>)
    {
        (f(std::get<Is>(fields_)), ...);
    }

    template <typename F, size_t... Is>
    void forEachConst_(F&& f, std::index_sequence<Is...>) const
    {
        (f(std::get<Is>(fields_)), ...);
    }

    void assertSameHostSize_() const
    {
#ifndef NDEBUG
        if constexpr (numFields_ == 0) return;

        const size_t n = std::get<0>(fields_).hostSize();
        bool ok = true;

        forEachConst_([&](auto const& a)
        {
            ok = ok && (a.hostSize() == n);
        }, std::index_sequence_for<Ts...>{});

        assert(ok && "SoA1D: host sizes mismatch between fields!");
#endif
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    SoA1D() = default;
    ~SoA1D() = default;

    SoA1D(const SoA1D&) = delete;
    SoA1D& operator=(const SoA1D&) = delete;

    SoA1D(SoA1D&&) noexcept = default;
    SoA1D& operator=(SoA1D&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const
    {
        if constexpr (numFields_ == 0) return 0;
        return std::get<0>(fields_).hostSize();
    }

public:
    // ---------------------------------------------------------------------
    // Host operations (lock-step)
    // ---------------------------------------------------------------------
    void clearHost()
    {
        forEach_([&](auto& a)
        {
            a.clearHost();
        }, std::index_sequence_for<Ts...>{});
    }

    void reserveHost(const size_t n)
    {
        forEach_([&](auto& a)
        {
            a.reserveHost(n);
        }, std::index_sequence_for<Ts...>{});
    }

    void resizeHost(const size_t n)
    {
        forEach_([&](auto& a)
        {
            a.resizeHost(n);
        }, std::index_sequence_for<Ts...>{});
    }

    void eraseHost(const size_t index)
    {
        forEach_([&](auto& a)
        {
            a.eraseHost(index);
        }, std::index_sequence_for<Ts...>{});
    }

    void pushRow(const Ts&... values)
    {
        (std::get<HostDeviceArray1D<Ts>>(fields_).pushHost(values), ...);
        assertSameHostSize_();
    }

    void insertRow(const size_t index, const Ts&... values)
    {
        (std::get<HostDeviceArray1D<Ts>>(fields_).insertHost(index, values), ...);
        assertSameHostSize_();
    }

public:
    // ---------------------------------------------------------------------
    // Transfers (lock-step)
    // ---------------------------------------------------------------------
    void copyHostToDevice(cudaStream_t stream)
    {
        assertSameHostSize_();

        forEach_([&](auto& a)
        {
            a.copyHostToDevice(stream);
        }, std::index_sequence_for<Ts...>{});
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        forEach_([&](auto& a)
        {
            a.copyDeviceToHost(stream);
        }, std::index_sequence_for<Ts...>{});
    }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    // Allocate an empty device buffer of size n for each field.
    // This does NOT touch host data (no copy back).
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        forEach_([&](auto& a)
        {
            a.allocateDevice(n, stream, zeroFill);
        }, std::index_sequence_for<Ts...>{});
    }

public:
    // ---------------------------------------------------------------------
    // Field access
    // ---------------------------------------------------------------------
    template <size_t I>
    auto* devicePtr()
    {
        static_assert(I < numFields_, "SoA1D::devicePtr<I> out of range");
        return std::get<I>(fields_).d_ptr;
    }

    template <size_t I>
    const auto* devicePtr() const
    {
        static_assert(I < numFields_, "SoA1D::devicePtr<I> out of range");
        return std::get<I>(fields_).d_ptr;
    }

    template <size_t I>
    auto hostCopy()
    {
        static_assert(I < numFields_, "SoA1D::hostCopy<I> out of range");
        return std::get<I>(fields_).getHostCopy();
    }

    template <size_t I>
    HostDeviceArray1D<std::tuple_element_t<I, std::tuple<Ts...>>>& field()
    {
        static_assert(I < numFields_, "SoA1D::field<I> out of range");
        return std::get<I>(fields_);
    }

    template <size_t I>
    const HostDeviceArray1D<std::tuple_element_t<I, std::tuple<Ts...>>>& field() const
    {
        static_assert(I < numFields_, "SoA1D::field<I> out of range");
        return std::get<I>(fields_);
    }
};