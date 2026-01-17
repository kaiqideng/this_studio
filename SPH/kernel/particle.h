#pragma once
#include "myUtility/myHostDeviceArray.h"
#include <algorithm>

struct WCSPH
{
private:
    // ---------------------------------------------------------------------
    // Fields
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> acceleration_;
    HostDeviceArray1D<double3> normal_;

    HostDeviceArray1D<double> density_;
    HostDeviceArray1D<double> pressure_;
    HostDeviceArray1D<double> initialDensity_;
    HostDeviceArray1D<double> smoothLength_;
    HostDeviceArray1D<double> mass_;
    HostDeviceArray1D<double> soundSpeed_;
    HostDeviceArray1D<double> viscosity_;

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
        const size_t n = hostSize_;
        bool ok = true;

        ok = ok && (position_.hostSize() == n);
        ok = ok && (velocity_.hostSize() == n);
        ok = ok && (acceleration_.hostSize() == n);
        ok = ok && (normal_.hostSize() == n);

        ok = ok && (density_.hostSize() == n);
        ok = ok && (pressure_.hostSize() == n);
        ok = ok && (initialDensity_.hostSize() == n);
        ok = ok && (smoothLength_.hostSize() == n);
        ok = ok && (mass_.hostSize() == n);
        ok = ok && (soundSpeed_.hostSize() == n);
        ok = ok && (viscosity_.hostSize() == n);

        ok = ok && (hashValue_.hostSize() == n);
        ok = ok && (hashIndex_.hostSize() == n);

        assert(ok && "WCSPH: field host sizes mismatch!");
#endif
    }

    void updateGridDim_()
    {
        if (blockDim_ == 0) { gridDim_ = 0; return; }
        gridDim_ = (deviceSize_ + blockDim_ - 1) / blockDim_;
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
        updateGridDim_();
    }

    size_t blockDim() const { return blockDim_; }
    size_t gridDim() const { return gridDim_; }

public:
    // ---------------------------------------------------------------------
    // Host operations
    // ---------------------------------------------------------------------
    void eraseHost(std::vector<size_t> index)
    {
        std::sort(index.begin(), index.end(), std::greater<size_t>());
        index.erase(std::unique(index.begin(), index.end()), index.end());

        const size_t hostSize0 = hostSize_;
        for (size_t i = 0; i < index.size(); i++)
        {
            if (index[i] < hostSize0 && index[i] >= 0)
            {
                position_.eraseHost(index[i]);
                velocity_.eraseHost(index[i]);
                acceleration_.eraseHost(index[i]);
                normal_.eraseHost(index[i]);

                density_.eraseHost(index[i]);
                pressure_.eraseHost(index[i]);
                initialDensity_.eraseHost(index[i]);
                smoothLength_.eraseHost(index[i]);
                mass_.eraseHost(index[i]);
                soundSpeed_.eraseHost(index[i]);
                viscosity_.eraseHost(index[i]);

                hashValue_.eraseHost(index[i]);
                hashIndex_.eraseHost(index[i]);

                hostSize_--;
            }
        }
    }

    void addHost(const double3 pos,
    const double3 vel,
    const double rho,
    const double p,
    const double rho0,
    const double h,
    const double m,
    const double c,
    const double nu)
    {
        position_.pushHost(pos);
        velocity_.pushHost(vel);
        acceleration_.pushHost(make_double3(0.0, 0.0, 0.0));
        normal_.pushHost(make_double3(0.0, 0.0, 0.0));

        density_.pushHost(rho);
        pressure_.pushHost(p);
        initialDensity_.pushHost(rho0);
        smoothLength_.pushHost(h);
        mass_.pushHost(m);
        soundSpeed_.pushHost(c);
        viscosity_.pushHost(nu);

        hashValue_.pushHost(-1);
        hashIndex_.pushHost(-1);

        hostSize_++;
        assertAligned_();
    }

    void copyFromHost(const WCSPH& other)
    {
        hostSize_ = other.hostSize_;
        blockDim_ = other.blockDim_;

        position_.setHost(other.position_.hostRef());
        velocity_.setHost(other.velocity_.hostRef());
        acceleration_.setHost(other.acceleration_.hostRef());
        normal_.setHost(other.normal_.hostRef());

        density_.setHost(other.density_.hostRef());
        pressure_.setHost(other.pressure_.hostRef());
        initialDensity_.setHost(other.initialDensity_.hostRef());
        smoothLength_.setHost(other.smoothLength_.hostRef());
        mass_.setHost(other.mass_.hostRef());
        soundSpeed_.setHost(other.soundSpeed_.hostRef());
        viscosity_.setHost(other.viscosity_.hostRef());

        hashValue_.setHost(other.hashValue_.hostRef());
        hashIndex_.setHost(other.hashIndex_.hostRef());

        assertAligned_();
    }

public:
    // ---------------------------------------------------------------------
    // Transfers
    // ---------------------------------------------------------------------
    void copyHostToDevice(cudaStream_t stream)
    {
        assertAligned_();

        position_.copyHostToDevice(stream);
        velocity_.copyHostToDevice(stream);
        acceleration_.copyHostToDevice(stream);
        normal_.copyHostToDevice(stream);

        density_.copyHostToDevice(stream);
        pressure_.copyHostToDevice(stream);
        initialDensity_.copyHostToDevice(stream);
        smoothLength_.copyHostToDevice(stream);
        mass_.copyHostToDevice(stream);
        soundSpeed_.copyHostToDevice(stream);
        viscosity_.copyHostToDevice(stream);

        hashValue_.copyHostToDevice(stream);
        hashIndex_.copyHostToDevice(stream);

        const size_t n = hostSize_;
        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = hostSize_;
        updateGridDim_();
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        position_.copyDeviceToHost(stream);
        velocity_.copyDeviceToHost(stream);
        acceleration_.copyDeviceToHost(stream);
        normal_.copyDeviceToHost(stream);

        density_.copyDeviceToHost(stream);
        pressure_.copyDeviceToHost(stream);
        initialDensity_.copyDeviceToHost(stream);
        smoothLength_.copyDeviceToHost(stream);
        mass_.copyDeviceToHost(stream);
        soundSpeed_.copyDeviceToHost(stream);
        viscosity_.copyDeviceToHost(stream);

        hashValue_.copyDeviceToHost(stream);
        hashIndex_.copyDeviceToHost(stream);

        hostSize_ = deviceSize_;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double3* acceleration() { return acceleration_.d_ptr; }
    double3* normal() { return normal_.d_ptr; }

    double* density() { return density_.d_ptr; }
    double* pressure() { return pressure_.d_ptr; }

    double* initialDensity() { return initialDensity_.d_ptr; }
    double* smoothLength() { return smoothLength_.d_ptr; }
    double* mass() { return mass_.d_ptr; }
    double* soundSpeed() { return soundSpeed_.d_ptr; }
    double* viscosity() { return viscosity_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Device pointers (hash)
    // ---------------------------------------------------------------------
    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies
    // ---------------------------------------------------------------------
    std::vector<double3> positionHostCopy() { return position_.getHostCopy(); }
    std::vector<double3> velocityHostCopy() { return velocity_.getHostCopy(); }
    std::vector<double3> accelerationHostCopy() { return acceleration_.getHostCopy(); }
    std::vector<double3> normalHostCopy() { return normal_.getHostCopy(); }

    std::vector<double> densityHostCopy() { return density_.getHostCopy(); }
    std::vector<double> pressureHostCopy() { return pressure_.getHostCopy(); }

    std::vector<double> smoothLengthHostCopy() { return smoothLength_.getHostCopy(); }
};