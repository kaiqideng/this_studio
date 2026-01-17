#pragma once
#include "myUtility/myHostDeviceArray.h"

struct pair
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> objectPointed_;
    HostDeviceArray1D<int> objectPointing_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    pair() = default;
    ~pair() = default;

    pair(const pair&) = delete;
    pair& operator=(const pair&) = delete;

    pair(pair&&) noexcept = default;
    pair& operator=(pair&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        objectPointed_.allocateDevice(n, stream, zeroFill);
        objectPointing_.allocateDevice(n, stream, zeroFill);

        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(objectPointed_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(objectPointing_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* objectPointed() { return objectPointed_.d_ptr; }
    int* objectPointing() { return objectPointing_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> objectPointedHostCopy() { return objectPointed_.getHostCopy(); }
    std::vector<int> objectPointingHostCopy() { return objectPointing_.getHostCopy(); }
};

struct objectPointed
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> neighborCount_;
    HostDeviceArray1D<int> neighborPrefixSum_;

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
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        neighborCount_.allocateDevice(n, stream, zeroFill);
        neighborPrefixSum_.allocateDevice(n, stream, zeroFill);
        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* neighborCount() { return neighborCount_.d_ptr; }
    int* neighborPrefixSum() { return neighborPrefixSum_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> neighborCountHostCopy() { return neighborCount_.getHostCopy(); }
    std::vector<int> neighborPrefixSumHostCopy() { return neighborPrefixSum_.getHostCopy(); }

    size_t numNeighborPairs()
    {
        if (deviceSize_ == 0) return 0;

        int last = 0;
        CUDA_CHECK(cudaMemcpy(&last,
                              neighborPrefixSum_.d_ptr + (deviceSize_ - 1),
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

        return static_cast<size_t>(last);
    }
};

struct objectPointing
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> interactionStart_;
    HostDeviceArray1D<int> interactionEnd_;

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
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        interactionStart_.allocateDevice(n, stream, zeroFill);
        interactionEnd_.allocateDevice(n, stream, zeroFill);

        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(interactionStart_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(interactionEnd_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* interactionStart() { return interactionStart_.d_ptr; }
    int* interactionEnd() { return interactionEnd_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> interactionStartHostCopy() { return interactionStart_.getHostCopy(); }
    std::vector<int> interactionEndHostCopy() { return interactionEnd_.getHostCopy(); }
};