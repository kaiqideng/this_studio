#pragma once
#include "myUtility/myHostDeviceArray.h"

struct interaction
{
private:
    // ---------------------------------------------------------------------
    // Data (SoA)
    // ---------------------------------------------------------------------
    using Storage = SoA1D<int, int>;   // (objectPointed, objectPointing)
    Storage data_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    interaction() = default;
    ~interaction() = default;

    interaction(const interaction&) = delete;
    interaction& operator=(const interaction&) = delete;

    interaction(interaction&&) noexcept = default;
    interaction& operator=(interaction&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // Allocate an empty device buffer of size n (no host touch).
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        data_.allocateDevice(n, stream, zeroFill);
        CUDA_CHECK(cudaMemsetAsync(data_.devicePtr<0>(), 0xFF, n * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(data_.devicePtr<1>(), 0xFF, n * sizeof(int), stream));
        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* objectPointed() { return data_.devicePtr<0>(); }
    int* objectPointing() { return data_.devicePtr<1>(); }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> objectPointedHostCopy() { return data_.hostCopy<0>(); }
    std::vector<int> objectPointingHostCopy() { return data_.hostCopy<1>(); }
};

struct objectPointed
{
private:
    using Storage = SoA1D<int, int>;   // (neighborCount, neighborPrefixSum)
    Storage data_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    objectPointed() = default;
    ~objectPointed() = default;

    objectPointed(const objectPointed&) = delete;
    objectPointed& operator=(const objectPointed&) = delete;

    objectPointed(objectPointed&&) noexcept = default;
    objectPointed& operator=(objectPointed&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // Allocate an empty device buffer of size n (no host touch).
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        data_.allocateDevice(n, stream, zeroFill);
        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* neighborCount() { return data_.devicePtr<0>(); }
    int* neighborPrefixSum() { return data_.devicePtr<1>(); }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> neighborCountHostCopy() { return data_.hostCopy<0>(); }
    std::vector<int> neighborPrefixSumHostCopy() { return data_.hostCopy<1>(); }
    size_t numNeighborPairs()
    {
        int sum = 0;
        if (deviceSize_ > 0)
        {
            CUDA_CHECK(cudaMemcpy(&sum,
                                  neighborPrefixSum(),
                                  deviceSize_ * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
        return static_cast<size_t>(sum);
    }
};

struct objectPointing
{
private:
    using Storage = SoA1D<int, int>;   // (interactionStart, interactionEnd)
    Storage data_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    objectPointing() = default;
    ~objectPointing() = default;

    objectPointing(const objectPointing&) = delete;
    objectPointing& operator=(const objectPointing&) = delete;

    objectPointing(objectPointing&&) noexcept = default;
    objectPointing& operator=(objectPointing&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // Allocate an empty device buffer of size n (no host touch).
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        data_.allocateDevice(n, stream, zeroFill);
        CUDA_CHECK(cudaMemsetAsync(data_.devicePtr<0>(), 0xFF, n * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(data_.devicePtr<1>(), 0xFF, n * sizeof(int), stream));
        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* interactionStart() { return data_.devicePtr<0>(); }
    int* interactionEnd() { return data_.devicePtr<1>(); }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> interactionStartHostCopy() { return data_.hostCopy<0>(); }
    std::vector<int> interactionEndHostCopy() { return data_.hostCopy<1>(); }
};