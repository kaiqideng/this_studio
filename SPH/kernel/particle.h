#pragma once
#include "myUtility/myHostDeviceArray.h"
#include <algorithm>

struct WCSPH
{
private:
    // ---------------------------------------------------------------------
    // AoS (actually SoA storage):
    //   0 position
    //   1 velocity
    //   2 acceleration
    //   3 normal
    //   4 density
    //   5 pressure
    //   6 initialDensity
    //   7 smoothLength
    //   8 mass
    //   9 soundSpeed
    //   10 viscosity
    // ---------------------------------------------------------------------
    using Storage = SoA1D<
        double3, double3,
        double3, double3,
        double,  double,
        double,  double,
        double,  double,
        double
    >;

    Storage data_;

    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;

    size_t hostSize_ {0};
    size_t deviceSize_ {0};
    size_t blockDim_ {256};
    size_t gridDim_ {0};

private:
    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------
    void assertAligned_() const
    {
#ifndef NDEBUG
        const size_t n = hostSize();
        bool ok = true;

        ok = ok && (data_.hostSize() == n);
        ok = ok && (hashValue_.hostSize() == n);
        ok = ok && (hashIndex_.hostSize() == n);

        assert(ok && "WCSPH: field host sizes mismatch (data_/hash arrays)!");
#endif
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    WCSPH() = default;
    ~WCSPH() = default;

    WCSPH(const WCSPH&) = delete;
    WCSPH& operator=(const WCSPH&) = delete;

    WCSPH(WCSPH&&) noexcept = default;
    WCSPH& operator=(WCSPH&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const { return hostSize_; }
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // GPU computation parameters
    // ---------------------------------------------------------------------  
    void setBlockDim(const size_t blockDim) 
    { 
        if (blockDim == 0) return;
        blockDim_ = blockDim; 
    }

    size_t blockDim() const { return blockDim_; }
    size_t gridDim() const { return gridDim_; }

public:
    // ---------------------------------------------------------------------
    // Host operations
    // ---------------------------------------------------------------------
    void clearHost()
    {
        data_.clearHost();
        hashValue_.clearHost();
        hashIndex_.clearHost();

        hostSize_ = 0;
    }

    void eraseHost(std::vector<size_t> index)
    {
        std::sort(index.begin(), index.end(), std::greater<size_t>());
        index.erase(std::unique(index.begin(), index.end()), index.end());
        const size_t hostSize0 = hostSize_;
        for (size_t i = 0; i < index.size(); i++)
        {
            if (index[i] < hostSize0 && index[i] > 0)
            {
                data_.eraseHost(index[i]);
                hashValue_.eraseHost(index[i]);
                hashIndex_.eraseHost(index[i]);

                hostSize_--;
            }
        }
    }

    // Append SPH at the end.
    void addHost(const double3 pos,
                 const double3 vel,
                 const double  rho,
                 const double  p,
                 const double  rho0,
                 const double  h,
                 const double  m,
                 const double  c,
                 const double  nu)
    {
        data_.pushRow(pos, vel,
                      make_double3(0.0, 0.0, 0.0),     // acceleration
                      make_double3(0.0, 0.0, 0.0),     // normal
                      rho, p,
                      rho0, h,
                      m, c,
                      nu);

        // Keep hash arrays aligned (host-side default = -1).
        hashValue_.pushHost(-1);
        hashIndex_.pushHost(-1);

        deviceSize_++;
        assertAligned_();
    }

    void copyFromHost(const WCSPH& other)
    {
        // Copy segmentation on host
        hostSize_ = other.hostSize_;
        // Do NOT copy device sizes (keep current device state untouched)
        // SPHDeviceSize_ / dummyDeviceSize_ are left as-is.

        // Copy host vectors of all SoA fields
        data_.field<0>().setHost(other.data_.field<0>().hostRef());  // position
        data_.field<1>().setHost(other.data_.field<1>().hostRef());  // velocity
        data_.field<2>().setHost(other.data_.field<2>().hostRef());  // acceleration
        data_.field<3>().setHost(other.data_.field<3>().hostRef());  // normal

        data_.field<4>().setHost(other.data_.field<4>().hostRef());  // density
        data_.field<5>().setHost(other.data_.field<5>().hostRef());  // pressure

        data_.field<6>().setHost(other.data_.field<6>().hostRef());  // initialDensity
        data_.field<7>().setHost(other.data_.field<7>().hostRef());  // smoothLength
        data_.field<8>().setHost(other.data_.field<8>().hostRef());  // mass
        data_.field<9>().setHost(other.data_.field<9>().hostRef());  // soundSpeed
        data_.field<10>().setHost(other.data_.field<10>().hostRef()); // viscosity

        // Copy host vectors of hash arrays
        hashValue_.setHost(other.hashValue_.hostRef());
        hashIndex_.setHost(other.hashIndex_.hostRef());
    }

public:
    // ---------------------------------------------------------------------
    // Transfers
    // ---------------------------------------------------------------------
    void copyHostToDevice(cudaStream_t stream)
    {
        assertAligned_();

        data_.copyHostToDevice(stream);

        hashValue_.copyHostToDevice(stream);
        hashIndex_.copyHostToDevice(stream);

        // Force hash arrays to 0xFF on device (int = -1).
        const size_t n = hostSize_;
        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = hostSize_;
        if (blockDim_ > 0) gridDim_ = (deviceSize_ + blockDim_ - 1) / blockDim_;
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        data_.copyDeviceToHost(stream);
        hashValue_.copyDeviceToHost(stream);
        hashIndex_.copyDeviceToHost(stream);

        hostSize_ = deviceSize_;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers (data)
    // ---------------------------------------------------------------------
    double3* position() { return data_.devicePtr<0>(); }
    double3* velocity() { return data_.devicePtr<1>(); }
    double3* acceleration() { return data_.devicePtr<2>(); }
    double3* normal() { return data_.devicePtr<3>(); }

    double* density() { return data_.devicePtr<4>(); }
    double* pressure() { return data_.devicePtr<5>(); }

    double* initialDensity() { return data_.devicePtr<6>(); }
    double* smoothLength() { return data_.devicePtr<7>(); }
    double* mass() { return data_.devicePtr<8>(); }
    double* soundSpeed() { return data_.devicePtr<9>(); }
    double* viscosity() { return data_.devicePtr<10>(); }

public:
    // ---------------------------------------------------------------------
    // Device pointers (hash)
    // ---------------------------------------------------------------------
    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<double3> positionHostCopy() { return data_.hostCopy<0>(); }
    std::vector<double3> velocityHostCopy() { return data_.hostCopy<1>(); }
    std::vector<double3> accelerationHostCopy() { return data_.hostCopy<2>(); }
    std::vector<double3> normalHostCopy() { return data_.hostCopy<3>(); }

    std::vector<double> densityHostCopy() { return data_.hostCopy<4>(); }
    std::vector<double> pressureHostCopy() { return data_.hostCopy<5>(); }

    std::vector<double> smoothLengthHostCopy() { return data_.hostCopy<7>(); }
};