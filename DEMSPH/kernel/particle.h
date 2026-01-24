#pragma once
#include "myUtility/myHostDeviceArray.h"
#include "myUtility/myMat.h"
#include <cassert>

struct ball
{
private:
    // ---------------------------------------------------------------------
    // Fields
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;

    HostDeviceArray1D<double> radius_;
    HostDeviceArray1D<double> inverseMass_;

    HostDeviceArray1D<int> materialID_;
    HostDeviceArray1D<int> clumpID_;
    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;

    size_t hostSize_ {0};
    size_t deviceSize_ {0};
    size_t blockDim_ {1};
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
        ok = ok && (angularVelocity_.hostSize() == n);
        ok = ok && (force_.hostSize() == n);
        ok = ok && (torque_.hostSize() == n);

        ok = ok && (radius_.hostSize() == n);
        ok = ok && (inverseMass_.hostSize() == n);

        ok = ok && (materialID_.hostSize() == n);
        ok = ok && (clumpID_.hostSize() == n);
        ok = ok && (hashValue_.hostSize() == n);
        ok = ok && (hashIndex_.hostSize() == n);

        assert(ok && "ball: field host sizes mismatch!");
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
    ball() = default;
    ~ball() = default;

    ball(const ball&) = delete;
    ball& operator=(const ball&) = delete;

    ball(ball&&) noexcept = default;
    ball& operator=(ball&&) noexcept = default;

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
    void eraseHost(const size_t index)
    {
        if (index < hostSize_)
        {
            position_.eraseHost(index);
            velocity_.eraseHost(index);
            angularVelocity_.eraseHost(index);
            force_.eraseHost(index);
            torque_.eraseHost(index);

            radius_.eraseHost(index);
            inverseMass_.eraseHost(index);

            materialID_.eraseHost(index);
            clumpID_.eraseHost(index);
            hashValue_.eraseHost(index);
            hashIndex_.eraseHost(index);

            hostSize_--;
        }
    }

    void addHost(const double3 pos,
    const double3 vel,
    const double3 omega,
    const double3 f,
    const double3 tau,
    const double r,
    const double invM,
    const int matID,
    const int cID,
    const int hV,
    const int hI)
    {
        position_.pushHost(pos);
        velocity_.pushHost(vel);
        angularVelocity_.pushHost(omega);
        force_.pushHost(f);
        torque_.pushHost(tau);

        radius_.pushHost(r);
        inverseMass_.pushHost(invM);

        materialID_.pushHost(matID);
        clumpID_.pushHost(cID);
        hashValue_.pushHost(hV);
        hashIndex_.pushHost(hI);

        hostSize_++;
        assertAligned_();
    }

    void copyFromHost(const ball& other)
    {
        hostSize_ = other.hostSize_;
        blockDim_ = other.blockDim_;

        position_.setHost(other.position_.hostRef());
        velocity_.setHost(other.velocity_.hostRef());
        angularVelocity_.setHost(other.angularVelocity_.hostRef());
        force_.setHost(other.force_.hostRef());
        torque_.setHost(other.torque_.hostRef());

        radius_.setHost(other.radius_.hostRef());
        inverseMass_.setHost(other.inverseMass_.hostRef());

        materialID_.setHost(other.materialID_.hostRef());
        clumpID_.setHost(other.clumpID_.hostRef());
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
        angularVelocity_.copyHostToDevice(stream);
        force_.copyHostToDevice(stream);
        torque_.copyHostToDevice(stream);

        radius_.copyHostToDevice(stream);
        inverseMass_.copyHostToDevice(stream);

        materialID_.copyHostToDevice(stream);
        clumpID_.copyHostToDevice(stream);

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
        angularVelocity_.copyDeviceToHost(stream);
        force_.copyDeviceToHost(stream);
        torque_.copyDeviceToHost(stream);

        radius_.copyDeviceToHost(stream);
        inverseMass_.copyDeviceToHost(stream);

        materialID_.copyDeviceToHost(stream);
        clumpID_.copyDeviceToHost(stream);
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
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    double3* force() { return force_.d_ptr; }
    double3* torque() { return torque_.d_ptr; }

    double* radius() { return radius_.d_ptr; }
    double* inverseMass() { return inverseMass_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Device pointers (IDs / hash)
    // ---------------------------------------------------------------------
    int* materialID() { return materialID_.d_ptr; }
    int* clumpID() { return clumpID_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies
    // ---------------------------------------------------------------------
    std::vector<double3> positionHostCopy() { return position_.getHostCopy(); }
    std::vector<double3> velocityHostCopy() { return velocity_.getHostCopy(); }
    std::vector<double3> angularVelocityHostCopy() { return angularVelocity_.getHostCopy(); }
    std::vector<double3> forceHostCopy() { return force_.getHostCopy(); }
    std::vector<double3> torqueHostCopy() { return torque_.getHostCopy(); }

    const std::vector<double>& radiusHostRef() { return radius_.hostRef(); }
    const std::vector<double>& inverseMassHostRef() { return inverseMass_.hostRef(); }
    const std::vector<int>& materialIDHostRef() { return materialID_.hostRef(); }
    const std::vector<int>& clumpIDHostRef() { return clumpID_.hostRef(); }

    void setClumpID(const std::vector<int>& c)
    {
        if (c.size() != hostSize_) return;
        clumpID_.setHost(c);
    }
};

struct clump
{
private:
    // ---------------------------------------------------------------------
    // Fields
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;

    HostDeviceArray1D<quaternion> orientation_;
    HostDeviceArray1D<symMatrix> inverseInertiaTensor_;

    HostDeviceArray1D<double> inverseMass_;

    HostDeviceArray1D<int> materialID_;
    HostDeviceArray1D<int> pebbleStart_;
    HostDeviceArray1D<int> pebbleEnd_;

    size_t hostSize_ {0};
    size_t deviceSize_ {0};
    size_t blockDim_ {1};
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
        ok = ok && (angularVelocity_.hostSize() == n);
        ok = ok && (force_.hostSize() == n);
        ok = ok && (torque_.hostSize() == n);

        ok = ok && (orientation_.hostSize() == n);
        ok = ok && (inverseInertiaTensor_.hostSize() == n);

        ok = ok && (inverseMass_.hostSize() == n);

        ok = ok && (materialID_.hostSize() == n);
        ok = ok && (pebbleStart_.hostSize() == n);
        ok = ok && (pebbleEnd_.hostSize() == n);

        assert(ok && "clump: field host sizes mismatch!");
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
    clump() = default;
    ~clump() = default;

    clump(const clump&) = delete;
    clump& operator=(const clump&) = delete;

    clump(clump&&) noexcept = default;
    clump& operator=(clump&&) noexcept = default;

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
    void eraseHost(const size_t index)
    {
        if (index < hostSize_)
        {
            position_.eraseHost(index);
            velocity_.eraseHost(index);
            angularVelocity_.eraseHost(index);
            force_.eraseHost(index);
            torque_.eraseHost(index);

            orientation_.eraseHost(index);
            inverseInertiaTensor_.eraseHost(index);

            inverseMass_.eraseHost(index);

            materialID_.eraseHost(index);
            pebbleStart_.eraseHost(index);
            pebbleEnd_.eraseHost(index);

            hostSize_--;
        }
    }

    void addHost(const double3 pos,
    const double3 vel,
    const double3 omega,
    const double3 f,
    const double3 tau,
    const quaternion q,
    const symMatrix invI,
    const double invM,
    const int matID,
    const int pebbleS,
    const int pebbleE)
    {
        position_.pushHost(pos);
        velocity_.pushHost(vel);
        angularVelocity_.pushHost(omega);
        force_.pushHost(f);
        torque_.pushHost(tau);

        orientation_.pushHost(q);
        inverseInertiaTensor_.pushHost(invI);

        inverseMass_.pushHost(invM);

        materialID_.pushHost(matID);
        pebbleStart_.pushHost(pebbleS);
        pebbleEnd_.pushHost(pebbleE);

        hostSize_++;
        assertAligned_();
    }

    void copyFromHost(const clump& other)
    {
        hostSize_ = other.hostSize_;
        blockDim_ = other.blockDim_;

        position_.setHost(other.position_.hostRef());
        velocity_.setHost(other.velocity_.hostRef());
        angularVelocity_.setHost(other.angularVelocity_.hostRef());
        force_.setHost(other.force_.hostRef());
        torque_.setHost(other.torque_.hostRef());

        orientation_.setHost(other.orientation_.hostRef());
        inverseInertiaTensor_.setHost(other.inverseInertiaTensor_.hostRef());

        inverseMass_.setHost(other.inverseMass_.hostRef());

        materialID_.setHost(other.materialID_.hostRef());
        pebbleStart_.setHost(other.pebbleStart_.hostRef());
        pebbleEnd_.setHost(other.pebbleEnd_.hostRef());

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
        angularVelocity_.copyHostToDevice(stream);
        force_.copyHostToDevice(stream);
        torque_.copyHostToDevice(stream);

        orientation_.copyHostToDevice(stream);
        inverseInertiaTensor_.copyHostToDevice(stream);

        inverseMass_.copyHostToDevice(stream);

        materialID_.copyHostToDevice(stream);
        pebbleStart_.copyHostToDevice(stream);
        pebbleEnd_.copyHostToDevice(stream);

        deviceSize_ = hostSize_;
        updateGridDim_();
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        position_.copyDeviceToHost(stream);
        velocity_.copyDeviceToHost(stream);
        angularVelocity_.copyDeviceToHost(stream);
        force_.copyDeviceToHost(stream);
        torque_.copyDeviceToHost(stream);

        orientation_.copyDeviceToHost(stream);
        inverseInertiaTensor_.copyDeviceToHost(stream);

        inverseMass_.copyDeviceToHost(stream);

        materialID_.copyDeviceToHost(stream);
        pebbleStart_.copyDeviceToHost(stream);
        pebbleEnd_.copyDeviceToHost(stream);

        hostSize_ = deviceSize_;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    double3* force() { return force_.d_ptr; }
    double3* torque() { return torque_.d_ptr; }

    quaternion* orientation() { return orientation_.d_ptr; }
    symMatrix* inverseInertiaTensor() { return inverseInertiaTensor_.d_ptr; }

    double* inverseMass() { return inverseMass_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Device pointers (IDs / ranges)
    // ---------------------------------------------------------------------
    int* materialID() { return materialID_.d_ptr; }
    int* pebbleStart() { return pebbleStart_.d_ptr; }
    int* pebbleEnd() { return pebbleEnd_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies
    // ---------------------------------------------------------------------
    std::vector<double3> positionHostCopy() { return position_.getHostCopy(); }
    std::vector<double3> velocityHostCopy() { return velocity_.getHostCopy(); }
    std::vector<double3> angularVelocityHostCopy() { return angularVelocity_.getHostCopy(); }
    std::vector<double3> forceHostCopy() { return force_.getHostCopy(); }
    std::vector<double3> torqueHostCopy() { return torque_.getHostCopy(); }

    const std::vector<quaternion>& orientationHostRef() { return orientation_.hostRef(); }
    const std::vector<symMatrix>& inverseInertiaTensorHostRef() { return inverseInertiaTensor_.hostRef(); }

    const std::vector<double>& inverseMassHostRef() { return inverseMass_.hostRef(); }

    const std::vector<int>& materialIDHostRef() { return materialID_.hostRef(); }
    const std::vector<int>& pebbleStartHostRef() { return pebbleStart_.hostRef(); }
    const std::vector<int>& pebbleEndHostRef() { return pebbleEnd_.hostRef(); }

    void setPebbleStartHost(const std::vector<int>& s)
    {
        if (s.size() != hostSize_) return;
        pebbleStart_.setHost(s);
    }

    void setPebbleEndHost(const std::vector<int>& e)
    {
        if (e.size() != hostSize_) return;
        pebbleEnd_.setHost(e);
    }
};

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

    HostDeviceArray1D<double> densityChange_;
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
    size_t blockDim_ {1};
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

        ok = ok && (densityChange_.hostSize() == n);
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
    void eraseHost(const size_t index)
    {
        if (index < hostSize_)
        {
            position_.eraseHost(index);
            velocity_.eraseHost(index);
            acceleration_.eraseHost(index);
            normal_.eraseHost(index);

            densityChange_.eraseHost(index);
            density_.eraseHost(index);
            pressure_.eraseHost(index);
            initialDensity_.eraseHost(index);
            smoothLength_.eraseHost(index);
            mass_.eraseHost(index);
            soundSpeed_.eraseHost(index);
            viscosity_.eraseHost(index);

            hashValue_.eraseHost(index);
            hashIndex_.eraseHost(index);

            hostSize_--;
        }
    }

    void addHost(const double3 pos,
    const double3 vel,
    const double3 acc,
    const double3 n,
    const double dRho,
    const double rho,
    const double p,
    const double rho0,
    const double h,
    const double m,
    const double c,
    const double nu,
    const int hV,
    const int hI)
    {
        position_.pushHost(pos);
        velocity_.pushHost(vel);
        acceleration_.pushHost(acc);
        normal_.pushHost(n);

        densityChange_.pushHost(dRho);
        density_.pushHost(rho);
        pressure_.pushHost(p);
        initialDensity_.pushHost(rho0);
        smoothLength_.pushHost(h);
        mass_.pushHost(m);
        soundSpeed_.pushHost(c);
        viscosity_.pushHost(nu);

        hashValue_.pushHost(hV);
        hashIndex_.pushHost(hI);

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

        densityChange_.setHost(other.densityChange_.hostRef());
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

        densityChange_.copyHostToDevice(stream);
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

        densityChange_.copyDeviceToHost(stream);
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

    double* densityChange() { return densityChange_.d_ptr; }
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

    const std::vector<double>& smoothLengthHostRef() { return smoothLength_.hostRef(); }
};