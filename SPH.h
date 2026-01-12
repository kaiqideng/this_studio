#pragma once
#include "DEM/kernel/cudaKernel/myStruct/myUtility/myHostDeviceArray1D.h"

struct WCSPH
{
private:
    // non-constant per-particle data (host <-> device)
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double> density_;
    HostDeviceArray1D<double> pressure_;

    // constant per-particle data (host -> device only)
    constantHostDeviceArray1D<double> soundSpeed_;
    constantHostDeviceArray1D<double> mass_;
    constantHostDeviceArray1D<double> initialDensity_;
    constantHostDeviceArray1D<double> smoothLength_;
    constantHostDeviceArray1D<double> kinematicViscosity_;

    // device-only
    DeviceArray1D<int> hashIndex_;
    DeviceArray1D<int> hashValue_;

    DeviceArray1D<double3> acc_;
    DeviceArray1D<double3> normal_;

    size_t SPHHostSize_ {0};
    size_t SPHDeviceSize_ {0};
    size_t dummyHostSize_ {0};
    size_t dummyDeviceSize_ {0};

public:
    WCSPH() = default;
    ~WCSPH() = default;
    WCSPH(const WCSPH&) = delete;
    WCSPH& operator=(const WCSPH&) = delete;
    WCSPH(WCSPH&&) noexcept = default;
    WCSPH& operator=(WCSPH&&) noexcept = default;

    size_t SPHHostSize() const { return SPHHostSize_; }
    size_t SPHDeviceSize() const { return SPHDeviceSize_; }
    size_t dummyHostSize() const { return dummyHostSize_; }
    size_t dummyDeviceSize() const { return dummyDeviceSize_; }

    void addSPHHost(const double3 pos,
    const double3 vel,
    const double c,
    const double m,
    const double rho0,
    const double h,
    const double nu)
    {
        size_t n = SPHHostSize_;

        position_.insertHostData(n, pos);
        velocity_.insertHostData(n, vel);

        soundSpeed_.insertHostData(n, c);
        mass_.insertHostData(n, m);
        initialDensity_.insertHostData(n, rho0);
        density_.insertHostData(n, rho0);
        smoothLength_.insertHostData(n, h);
        kinematicViscosity_.insertHostData(n, nu);

        SPHHostSize_++;
    }

    void addDummyHost(const double3 pos,
    const double3 vel,
    const double c,
    const double m,
    const double rho0,
    const double h,
    const double nu)
    {
        size_t n = SPHHostSize_ + dummyHostSize_; 

        position_.insertHostData(n, pos);
        velocity_.insertHostData(n, vel);

        soundSpeed_.insertHostData(n, c);
        mass_.insertHostData(n, m);
        initialDensity_.insertHostData(n, rho0);
        density_.insertHostData(n, rho0);
        smoothLength_.insertHostData(n, h);
        kinematicViscosity_.insertHostData(n, nu);

        dummyHostSize_++;
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        density_.download(stream);

        soundSpeed_.download(stream);
        mass_.download(stream);
        initialDensity_.download(stream);
        smoothLength_.download(stream);
        kinematicViscosity_.download(stream);

        // device-only arrays sized according to number of particles
        size_t n = SPHHostSize_ + dummyHostSize_;
        hashIndex_.allocDeviceArray(n, stream);
        hashValue_.allocDeviceArray(n, stream);
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, n * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, n * sizeof(int), stream));

        acc_.allocDeviceArray(n, stream);
        normal_.allocDeviceArray(n, stream);

        SPHDeviceSize_ = SPHHostSize_;
        dummyDeviceSize_ = dummyHostSize_;
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        density_.upload(stream);

        SPHHostSize_ = SPHDeviceSize_;
        dummyHostSize_ = dummyDeviceSize_;
    }

    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double* density() { return density_.d_ptr; }
    double* pressure() { return pressure_.d_ptr; }

    const double* soundSpeed() { return soundSpeed_.d_ptr; }
    const double* mass() { return mass_.d_ptr; }
    const double* initialDensity() { return initialDensity_.d_ptr; }
    const double* smoothLength() { return smoothLength_.d_ptr; }
    const double* kinematicViscosity() { return kinematicViscosity_.d_ptr; }

    int* hashIndex() { return hashIndex_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }

    double3* acceleration() { return acc_.d_ptr; }
    double3* normal() { return normal_.d_ptr; }

    std::vector<double3> positionVector() { return position_.getHostData(); }
    std::vector<double3> velocityVector() { return velocity_.getHostData(); }
    std::vector<double> densityVector() { return density_.getHostData(); }
    std::vector<double> pressureVector() { return pressure_.getHostData(); }

    const std::vector<double>& soundSpeedVector() const
    {
        return soundSpeed_.getHostData();
    }
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

struct ISPH
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
    ISPH() = default;
    ~ISPH() = default;
    ISPH(const ISPH&) = delete;
    ISPH& operator=(const ISPH&) = delete;
    ISPH(ISPH&&) noexcept = default;
    ISPH& operator=(ISPH&&) noexcept = default;

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

struct SPHInteraction
{
private:
    HostDeviceArray1D<int> objectPointed_;
    HostDeviceArray1D<int> objectPointing_;
    HostDeviceArray1D<double3> force_;

    DeviceArray1D<double3> gradientKernel_;
    DeviceArray1D<double3> gradientKernelStar_;

    size_t activeSize_ {0};

public:
    SPHInteraction() = default;
    ~SPHInteraction() = default;
    SPHInteraction(const SPHInteraction&) = delete;
    SPHInteraction& operator=(const SPHInteraction&) = delete;
    SPHInteraction(SPHInteraction&&) noexcept = default;
    SPHInteraction& operator=(SPHInteraction&&) noexcept = default;

    size_t activeSize() const { return activeSize_; }

    void alloc(size_t n, cudaStream_t stream)
    {
        objectPointed_.allocDeviceArray(n, stream);
        objectPointing_.allocDeviceArray(n, stream);
        force_.allocDeviceArray(n, stream);
        gradientKernel_.allocDeviceArray(n, stream);
        gradientKernelStar_.allocDeviceArray(n, stream);
    }

    void setActiveSize(size_t n, cudaStream_t stream)
    {
        activeSize_ = n;
        if (n > objectPointed_.deviceSize()) { alloc(n, stream); }
    }

    int* objectPointed() { return objectPointed_.d_ptr; }
    int* objectPointing() { return objectPointing_.d_ptr; }
    double3* force() { return force_.d_ptr; }

    double3* gradientKernel() { return gradientKernel_.d_ptr; }
    double3* gradientKernelStar() { return gradientKernelStar_.d_ptr; }

    std::vector<int> objectPointedVector() { return objectPointed_.getHostData(); }
    std::vector<int> objectPointingVector() { return objectPointing_.getHostData(); }
    std::vector<double3> forceVector() { return force_.getHostData(); }
};