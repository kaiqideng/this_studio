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
    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        angularVelocity_.upload(stream);
        force_.upload(stream);
        torque_.upload(stream);

        radius_.upload(stream);
        inverseMass_.upload(stream);
        materialID_.upload(stream);
        clumpID_.upload(stream);

        // device-only arrays sized according to number of particles
        const size_t n = deviceSize();
        hashIndex_.allocDeviceArray(n, stream);
        hashValue_.allocDeviceArray(n, stream);
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, hashValue_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, hashIndex_.deviceSize() * sizeof(int), stream));
    }

    // D2H: only for non-constant fields
    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        angularVelocity_.download(stream);
        force_.download(stream);
        torque_.download(stream);
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
        force_.upload(s);
    }

    void setTorqueVector(const std::vector<double3> T, cudaStream_t s)
    {
        if(T.size() != hostSize()) return;
        torque_.setHostData(T);
        torque_.upload(s);
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

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        angularVelocity_.upload(stream);
        force_.upload(stream);
        torque_.upload(stream);
        orientation_.upload(stream);

        inverseInertiaTensor_.upload(stream);
        inverseMass_.upload(stream);
        materialID_.upload(stream);
        pebbleStart_.upload(stream);
        pebbleEnd_.upload(stream);
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        angularVelocity_.download(stream);
        force_.download(stream);
        torque_.download(stream);
        orientation_.download(stream);
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
    std::vector<quaternion> orientationVector() { return orientation_.getHostData(); }

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