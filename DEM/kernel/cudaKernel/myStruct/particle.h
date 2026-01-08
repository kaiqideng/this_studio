#pragma once
#include "myUtility/myHostDeviceArray1D.h"

struct ball
{
private:
    // host + device, mutable
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;

    // host + device, constant (only H2D)
    constantHostDeviceArray1D<double> radius_;
    constantHostDeviceArray1D<double> inverseMass_;
    constantHostDeviceArray1D<int> materialID_;
    constantHostDeviceArray1D<int> clumpID_;

    // device-only
    DeviceArray1D<int> hashIndex_;
    DeviceArray1D<int> hashValue_;

public:
    ball() = default;
    ~ball() = default;
    ball(const ball&) = delete;
    ball& operator=(const ball&) = delete;
    ball(ball&&) noexcept = default;
    ball& operator=(ball&&) noexcept = default;

    // sizes
    size_t hostSize() const  { return position_.hostSize(); }
    size_t deviceSize() const{ return position_.deviceSize(); }

    // add one particle on host
    void addHost(const double3 pos,
    const double3 vel,
    const double3 angVel,
    const double3 F,
    const double3 T,
    const double r,
    const double invM,
    const int matID,
    const int clumpID_val)
    {
        position_.addHostData(pos);
        velocity_.addHostData(vel);
        angularVelocity_.addHostData(angVel);
        force_.addHostData(F);
        torque_.addHostData(T);

        radius_.addHostData(r);
        inverseMass_.addHostData(invM);
        materialID_.addHostData(matID);
        clumpID_.addHostData(clumpID_val);
    }

    void removeHost(size_t index)
    {
        position_.removeHostData(index);
        velocity_.removeHostData(index);
        angularVelocity_.removeHostData(index);
        force_.removeHostData(index);
        torque_.removeHostData(index);

        radius_.removeHostData(index);
        inverseMass_.removeHostData(index);
        materialID_.removeHostData(index);
        clumpID_.removeHostData(index);
    }

    void clearHost()
    {
        position_.clearHostData();
        velocity_.clearHostData();
        angularVelocity_.clearHostData();
        force_.clearHostData();
        torque_.clearHostData();

        radius_.clearHostData();
        inverseMass_.clearHostData();
        materialID_.clearHostData();
        clumpID_.clearHostData();
    }

    // H2D: allocate/reallocate device arrays and copy host -> device
    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        angularVelocity_.download(stream);
        force_.download(stream);
        torque_.download(stream);

        radius_.download(stream);
        inverseMass_.download(stream);
        materialID_.download(stream);
        clumpID_.download(stream);

        // device-only arrays sized according to number of particles
        const size_t n = deviceSize();
        hashIndex_.allocDeviceArray(n, stream);
        hashValue_.allocDeviceArray(n, stream);
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, hashValue_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, hashIndex_.deviceSize() * sizeof(int), stream));
    }

    // D2H: only for non-constant fields
    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        angularVelocity_.upload(stream);
        force_.upload(stream);
        torque_.upload(stream);
    }

    // device pointers accessors
    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    double3* force() { return force_.d_ptr; }
    double3* torque() { return torque_.d_ptr; }

    const double* radius() const { return radius_.d_ptr; }
    const double* inverseMass() const { return inverseMass_.d_ptr; }
    const int* materialID() const { return materialID_.d_ptr; }
    const int* clumpID() const { return clumpID_.d_ptr; }

    int* hashIndex() { return hashIndex_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }

    // host vectors accessors
    std::vector<double3> positionVector() { return position_.getHostData(); }
    std::vector<double3> velocityVector() { return velocity_.getHostData(); }
    std::vector<double3> angularVelocityVector() { return angularVelocity_.getHostData(); }
    std::vector<double3> forceVector() { return force_.getHostData(); }
    std::vector<double3> torqueVector() { return torque_.getHostData(); }

    const std::vector<double>& radiusVector() const { return radius_.getHostData(); }
    const std::vector<double>& inverseMassVector() const { return inverseMass_.getHostData(); }
    const std::vector<int>& materialIDVector() const { return materialID_.getHostData(); }
    const std::vector<int>& clumpIDVector() const { return clumpID_.getHostData(); }

    void setForceVector(const std::vector<double3> F, cudaStream_t s)
    {
        if(F.size() != hostSize()) return;
        force_.setHostData(F);
        force_.download(s);
    }

    void setTorqueVector(const std::vector<double3> T, cudaStream_t s)
    {
        if(T.size() != hostSize()) return;
        torque_.setHostData(T);
        torque_.download(s);
    }
};

struct clump
{
private:
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;
    HostDeviceArray1D<quaternion> orientation_;

    constantHostDeviceArray1D<symMatrix> inverseInertiaTensor_;
    constantHostDeviceArray1D<double> inverseMass_;
    constantHostDeviceArray1D<int> materialID_;
    constantHostDeviceArray1D<int> pebbleStart_;
    constantHostDeviceArray1D<int> pebbleEnd_;

public:
    clump() = default;
    ~clump() = default;
    clump(const clump&) = delete;
    clump& operator=(const clump&) = delete;
    clump(clump&&) noexcept = default;
    clump& operator=(clump&&) noexcept = default;

    size_t hostSize() const  { return position_.hostSize(); }
    size_t deviceSize() const{ return position_.deviceSize(); }

    void addHost(const double3 pos,
    const double3 vel,
    const double3 angVel,
    const double3 F,
    const double3 T,
    const quaternion q,
    const symMatrix invInertia,
    const double invMass,
    const int matID,
    const int pebbleStartIdx,
    const int pebbleEndIdx)
    {
        position_.addHostData(pos);
        velocity_.addHostData(vel);
        angularVelocity_.addHostData(angVel);
        force_.addHostData(F);
        torque_.addHostData(T);
        orientation_.addHostData(q);

        inverseInertiaTensor_.addHostData(invInertia);
        inverseMass_.addHostData(invMass);
        materialID_.addHostData(matID);
        pebbleStart_.addHostData(pebbleStartIdx);
        pebbleEnd_.addHostData(pebbleEndIdx);
    }

    void removeHost(size_t index)
    {
        position_.removeHostData(index);
        velocity_.removeHostData(index);
        angularVelocity_.removeHostData(index);
        force_.removeHostData(index);
        torque_.removeHostData(index);
        orientation_.removeHostData(index);

        inverseInertiaTensor_.removeHostData(index);
        inverseMass_.removeHostData(index);
        materialID_.removeHostData(index);
        pebbleStart_.removeHostData(index);
        pebbleEnd_.removeHostData(index);
    }

    void clearHost()
    {
        position_.clearHostData();
        velocity_.clearHostData();
        angularVelocity_.clearHostData();
        force_.clearHostData();
        torque_.clearHostData();
        orientation_.clearHostData();

        inverseInertiaTensor_.clearHostData();
        inverseMass_.clearHostData();
        materialID_.clearHostData();
        pebbleStart_.clearHostData();
        pebbleEnd_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        angularVelocity_.download(stream);
        force_.download(stream);
        torque_.download(stream);
        orientation_.download(stream);

        inverseInertiaTensor_.download(stream);
        inverseMass_.download(stream);
        materialID_.download(stream);
        pebbleStart_.download(stream);
        pebbleEnd_.download(stream);
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        angularVelocity_.upload(stream);
        force_.upload(stream);
        torque_.upload(stream);
        orientation_.upload(stream);
    }

    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    double3* force() { return force_.d_ptr; }
    double3* torque() { return torque_.d_ptr; }
    quaternion* orientation() { return orientation_.d_ptr; }

    const symMatrix* inverseInertiaTensor() const { return inverseInertiaTensor_.d_ptr; }
    const double* inverseMass() const { return inverseMass_.d_ptr; }
    const int* materialID() const { return materialID_.d_ptr; }
    const int* pebbleStart() const { return pebbleStart_.d_ptr; }
    const int* pebbleEnd() const { return pebbleEnd_.d_ptr; }

    std::vector<double3> positionVector() { return position_.getHostData(); }
    std::vector<double3> velocityVector() { return velocity_.getHostData(); }
    std::vector<double3> angularVelocityVector() { return angularVelocity_.getHostData(); }
    std::vector<double3> forceVector() { return force_.getHostData(); }
    std::vector<double3> torqueVector() { return torque_.getHostData(); }
    std::vector<quaternion> orientationVector() { return orientation_.getHostData();}

    const std::vector<symMatrix>& inverseInertiaTensorVector() const
    {
        return inverseInertiaTensor_.getHostData();
    }

    const std::vector<double>& inverseMassVector() const
    {
        return inverseMass_.getHostData();
    }

    const std::vector<int>& materialIDVector() const
    {
        return materialID_.getHostData();
    }

    const std::vector<int>& pebbleStartVector() const
    {
        return pebbleStart_.getHostData();
    }

    const std::vector<int>& pebbleEndVector() const
    {
        return pebbleEnd_.getHostData();
    }
};

struct SPH
{
private:
    // non-constant per-particle data (host <-> device)
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double>  pressure_;

    // constant per-particle data (host -> device only)
    constantHostDeviceArray1D<double> mass_;
    constantHostDeviceArray1D<double> initialDensity_;
    constantHostDeviceArray1D<double> smoothLength_;
    constantHostDeviceArray1D<double> kinematicViscosity_;

    // device-only
    DeviceArray1D<int> hashIndex_;
    DeviceArray1D<int> hashValue_;

    DeviceArray1D<double3> positionStar_;
    DeviceArray1D<double3> velocityStar_;
    DeviceArray1D<double> densityStar_;
    DeviceArray1D<double> pressureStar_;

    size_t SPHHostSize_ {0};
    size_t SPHDeviceSize_ {0};
    size_t ghostHostSize_ {0};
    size_t ghostDeviceSize_ {0};

public:
    SPH() = default;
    ~SPH() = default;
    SPH(const SPH&) = delete;
    SPH& operator=(const SPH&) = delete;
    SPH(SPH&&) noexcept = default;
    SPH& operator=(SPH&&) noexcept = default;

    size_t SPHHostSize() const { return SPHHostSize_; }
    size_t SPHDeviceSize() const { return SPHDeviceSize_; }
    size_t ghostHostSize() const { return ghostHostSize_; }
    size_t ghostDeviceSize() const { return ghostDeviceSize_; }

    void addSPHHost(const double3 pos,
    const double3 vel,
    const double p,
    const double m,
    const double rho0,
    const double h,
    const double nu)
    {
        size_t n = SPHHostSize_;

        position_.insertHostData(n, pos);
        velocity_.insertHostData(n, vel);
        pressure_.insertHostData(n, p);

        mass_.insertHostData(n, m);
        initialDensity_.insertHostData(n, rho0);
        smoothLength_.insertHostData(n, h);
        kinematicViscosity_.insertHostData(n, nu);

        SPHHostSize_++;
    }

    void addGhostHost(const double3 pos,
    const double3 vel,
    const double p,
    const double m,
    const double rho0,
    const double h,
    const double nu)
    {
        size_t n = SPHHostSize_ + ghostHostSize_; 

        position_.insertHostData(n, pos);
        velocity_.insertHostData(n, vel);
        pressure_.insertHostData(n, p);

        mass_.insertHostData(n, m);
        initialDensity_.insertHostData(n, rho0);
        smoothLength_.insertHostData(n, h);
        kinematicViscosity_.insertHostData(n, nu);

        ghostHostSize_++;
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        pressure_.download(stream);

        mass_.download(stream);
        initialDensity_.download(stream);
        smoothLength_.download(stream);
        kinematicViscosity_.download(stream);

        // device-only arrays sized according to number of particles
        size_t n = SPHHostSize_ + ghostHostSize_;
        hashIndex_.allocDeviceArray(n, stream);
        hashValue_.allocDeviceArray(n, stream);
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, n * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, n * sizeof(int), stream));

        positionStar_.allocDeviceArray(n, stream);
        velocityStar_.allocDeviceArray(n, stream);
        pressureStar_.allocDeviceArray(n, stream);
        densityStar_.allocDeviceArray(n, stream);

        SPHDeviceSize_ = SPHHostSize_;
        ghostDeviceSize_ = ghostHostSize_;
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        pressure_.upload(stream);

        SPHHostSize_ = SPHDeviceSize_;
        ghostHostSize_ = ghostDeviceSize_;
    }

    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double* pressure() { return pressure_.d_ptr; }

    const double* mass() { return mass_.d_ptr; }
    const double* initialDensity() { return initialDensity_.d_ptr; }
    const double* smoothLength() { return smoothLength_.d_ptr; }
    const double* kinematicViscosity() { return kinematicViscosity_.d_ptr; }

    int* hashIndex() { return hashIndex_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }

    double3* positionStar() { return positionStar_.d_ptr; }
    double3* velocityStar() { return velocityStar_.d_ptr; }
    double* pressureStar() { return pressureStar_.d_ptr; }
    double* densityStar() { return densityStar_.d_ptr; }

    std::vector<double3> positionVector() { return position_.getHostData(); }
    std::vector<double3> velocityVector() { return velocity_.getHostData(); }
    std::vector<double>  pressureVector() { return pressure_.getHostData(); }

    const std::vector<double>& massVector() const
    {
        return mass_.getHostData();
    }
    const std::vector<double>& initialDensityVector() const
    {
        return initialDensity_.getHostData();
    }
    const std::vector<double>& smoothLengthVector() const
    {
        return smoothLength_.getHostData();
    }
    const std::vector<double>& kinematicViscosityVector() const
    {
        return kinematicViscosity_.getHostData();
    }
};

struct virtualParticle
{
private:
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double> effectiveRadius_;

    DeviceArray1D<int> hashIndex_;
    DeviceArray1D<int> hashValue_;

public:
    virtualParticle() = default;
    ~virtualParticle() = default;
    virtualParticle(const virtualParticle&) = delete;
    virtualParticle& operator=(const virtualParticle&) = delete;
    virtualParticle(virtualParticle&&) noexcept = default;
    virtualParticle& operator=(virtualParticle&&) noexcept = default;

    size_t hostSize() const  { return position_.hostSize(); }
    size_t deviceSize() const{ return position_.deviceSize(); }

    void addHost(const double3 pos,
    const double r)
    {
        position_.addHostData(pos);
        effectiveRadius_.addHostData(r);
    }

    void removeHost(size_t index)
    {
        position_.removeHostData(index);
        effectiveRadius_.removeHostData(index);
    }

    void clearHost()
    {
        position_.clearHostData();
        effectiveRadius_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        effectiveRadius_.download(stream);

        const size_t n = deviceSize();
        hashIndex_.allocDeviceArray(n, stream);
        hashValue_.allocDeviceArray(n, stream);
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, hashValue_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, hashIndex_.deviceSize() * sizeof(int), stream));
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        effectiveRadius_.upload(stream);
    }

    double3* position() { return position_.d_ptr; }
    double* effectiveRadius() { return effectiveRadius_.d_ptr; }

    int* hashIndex() { return hashIndex_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }

    std::vector<double3> positionVector() { return position_.getHostData(); }
    std::vector<double> effectiveRadiusVector() { return effectiveRadius_.getHostData(); }
};