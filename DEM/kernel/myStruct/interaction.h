#pragma once
#include "myUtility/myHostDeviceArray1D.h"

struct solidInteraction
{
private:
    HostDeviceArray1D<int>     objectPointed_;
    HostDeviceArray1D<int>     objectPointing_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;
    HostDeviceArray1D<double3> contactPoint_;
    HostDeviceArray1D<double3> slidingSpring_;
    HostDeviceArray1D<double3> rollingSpring_;
    HostDeviceArray1D<double3> torsionSpring_;

    DeviceArray1D<int>         objectPointedHistory_;
    DeviceArray1D<int>         objectPointingHistory_;
    DeviceArray1D<double3>     slidingSpringHistory_;
    DeviceArray1D<double3>     rollingSpringHistory_;
    DeviceArray1D<double3>     torsionSpringHistory_;

    size_t activeSize_ {0};
    size_t activeSizeHistory_{0};

public:
    solidInteraction() = default;
    ~solidInteraction() = default;
    solidInteraction(const solidInteraction&) = delete;
    solidInteraction& operator=(const solidInteraction&) = delete;
    solidInteraction(solidInteraction&&) noexcept = default;
    solidInteraction& operator=(solidInteraction&&) noexcept = default;

    size_t deviceSize() const{ return objectPointed_.deviceSize(); }
    size_t activeSize() const{ return activeSize_; }

    void addHost(const int     objectPointed,
                 const int     objectPointing,
                 const double3 force,
                 const double3 torque,
                 const double3 contactPoint,
                 const double3 slidingSpring,
                 const double3 rollingSpring,
                 const double3 torsionSpring)
    {
        objectPointed_.addHostData(objectPointed);
        objectPointing_.addHostData(objectPointing);
        force_.addHostData(force);
        torque_.addHostData(torque);
        contactPoint_.addHostData(contactPoint);
        slidingSpring_.addHostData(slidingSpring);
        rollingSpring_.addHostData(rollingSpring);
        torsionSpring_.addHostData(torsionSpring);
    }

    void removeHost(size_t index)
    {
        objectPointed_.removeHostData(index);
        objectPointing_.removeHostData(index);
        force_.removeHostData(index);
        torque_.removeHostData(index);
        contactPoint_.removeHostData(index);
        slidingSpring_.removeHostData(index);
        rollingSpring_.removeHostData(index);
        torsionSpring_.removeHostData(index);
    }

    void clearHost()
    {
        objectPointed_.clearHostData();
        objectPointing_.clearHostData();
        force_.clearHostData();
        torque_.clearHostData();
        contactPoint_.clearHostData();
        slidingSpring_.clearHostData();
        rollingSpring_.clearHostData();
        torsionSpring_.clearHostData();
    }

    void alloc(size_t n, cudaStream_t stream)
    {
        objectPointed_.allocDeviceArray(n, stream);
        objectPointing_.allocDeviceArray(n, stream);
        force_.allocDeviceArray(n, stream);
        torque_.allocDeviceArray(n, stream);
        contactPoint_.allocDeviceArray(n, stream);
        slidingSpring_.allocDeviceArray(n, stream);
        rollingSpring_.allocDeviceArray(n, stream);
        torsionSpring_.allocDeviceArray(n, stream);
        objectPointedHistory_.allocDeviceArray(n, stream);
        objectPointingHistory_.allocDeviceArray(n, stream);
        slidingSpringHistory_.allocDeviceArray(n, stream);
        rollingSpringHistory_.allocDeviceArray(n, stream);
        torsionSpringHistory_.allocDeviceArray(n, stream);
    }

    void setActiveSize(size_t n, cudaStream_t stream)
    {
        activeSize_ = n;
        if(n > deviceSize())
        {
            objectPointed_.download(stream);
            objectPointing_.download(stream);
            force_.download(stream);
            torque_.download(stream);
            contactPoint_.download(stream);
            slidingSpring_.download(stream);
            rollingSpring_.download(stream);
            torsionSpring_.download(stream);
        }
    }

    void updateHistory(cudaStream_t stream)
    {
        if(activeSize_ > activeSizeHistory_)
        {
            objectPointedHistory_.allocDeviceArray(activeSize_, stream);
            objectPointingHistory_.allocDeviceArray(activeSize_, stream);
            slidingSpringHistory_.allocDeviceArray(activeSize_, stream);
            rollingSpringHistory_.allocDeviceArray(activeSize_, stream);
            torsionSpringHistory_.allocDeviceArray(activeSize_, stream);
        }
        if(activeSize_ > 0)
        {
            cuda_copy(objectPointedHistory(), objectPointed(), activeSize_,CopyDir::D2D, stream);
            cuda_copy(objectPointingHistory(), objectPointing(), activeSize_,CopyDir::D2D, stream);
            cuda_copy(slidingSpringHistory(), slidingSpring(), activeSize_,CopyDir::D2D, stream);
            cuda_copy(rollingSpringHistory(), rollingSpring(), activeSize_,CopyDir::D2D, stream);
            cuda_copy(torsionSpringHistory(), torsionSpring(), activeSize_,CopyDir::D2D, stream);
        }
        activeSizeHistory_ = activeSize_;
    }

    int*           objectPointed()        { return objectPointed_.d_ptr; }
    int*           objectPointing()       { return objectPointing_.d_ptr; }
    double3*       force()                { return force_.d_ptr; }
    double3*       torque()               { return torque_.d_ptr; }
    double3*       contactPoint()         { return contactPoint_.d_ptr; }
    double3*       slidingSpring()        { return slidingSpring_.d_ptr; }
    double3*       rollingSpring()        { return rollingSpring_.d_ptr; }
    double3*       torsionSpring()        { return torsionSpring_.d_ptr; }

    int*           objectPointedHistory()   { return objectPointedHistory_.d_ptr; }
    int*           objectPointingHistory()  { return objectPointingHistory_.d_ptr; }
    double3*       slidingSpringHistory()   { return slidingSpringHistory_.d_ptr; }
    double3*       rollingSpringHistory()   { return rollingSpringHistory_.d_ptr; }
    double3*       torsionSpringHistory()   { return torsionSpringHistory_.d_ptr; }

    std::vector<int>     objectPointedVector()   { return objectPointed_.getHostData(); }
    std::vector<int>     objectPointingVector()  { return objectPointing_.getHostData(); }
    std::vector<double3> forceVector()           { return force_.getHostData(); }
    std::vector<double3> torqueVector()          { return torque_.getHostData(); }
    std::vector<double3> contactPointVector()    { return contactPoint_.getHostData(); }
    std::vector<double3> slidingSpringVector()   { return slidingSpring_.getHostData(); }
    std::vector<double3> rollingSpringVector()   { return rollingSpring_.getHostData(); }
    std::vector<double3> torsionSpringVector()   { return torsionSpring_.getHostData(); }
};

struct bondedInteraction
{
private:
    HostDeviceArray1D<int>     objectPointed_;
    HostDeviceArray1D<int>     objectPointing_;
    HostDeviceArray1D<double>  normalForce_;
    HostDeviceArray1D<double>  torsionTorque_;
    HostDeviceArray1D<double3> shearForce_;
    HostDeviceArray1D<double3> bendingTorque_;
    HostDeviceArray1D<double3> contactNormal_;
    HostDeviceArray1D<int>     isBonded_;

public:
    bondedInteraction() = default;
    ~bondedInteraction() = default;
    bondedInteraction(const bondedInteraction&) = delete;
    bondedInteraction& operator=(const bondedInteraction&) = delete;
    bondedInteraction(bondedInteraction&&) noexcept = default;
    bondedInteraction& operator=(bondedInteraction&&) noexcept = default;

    size_t hostSize() const  { return objectPointed_.hostSize(); }
    size_t deviceSize() const{ return objectPointed_.deviceSize(); }
    
    void add(const std::vector<int> &ob0,
        const std::vector<int> &ob1,
        const std::vector<double3> &p,
        cudaStream_t stream)
    {
        if (ob0.size() != ob1.size()) return;

        upload(stream);

        std::vector<int> existingPointed  = objectPointed_.getHostData();
        std::vector<int> existingPointing = objectPointing_.getHostData();

        for (size_t i = 0; i < ob0.size(); ++i)
        {
            int i0 = ob0[i];
            int i1 = ob1[i];

            if (i0 < 0 || i1 < 0) continue;
            if (static_cast<size_t>(i0) >= p.size() ||
                static_cast<size_t>(i1) >= p.size()) continue;
            if (i0 == i1) continue;

            int a = i0;
            int b = i1;
            if (a > b) std::swap(a, b);

            bool found = false;
            for (size_t j = 0; j < existingPointed.size(); ++j)
            {
                if (existingPointed[j] == a && existingPointing[j] == b)
                {
                    found = true;
                    break;
                }
            }
            if (found) continue;

            existingPointed.push_back(a);
            existingPointing.push_back(b);

            double3 n      = p[a] - p[b];
            double3 n_norm = normalize(n);

            objectPointed_.addHostData(a);
            objectPointing_.addHostData(b);
            contactNormal_.addHostData(n_norm);

            shearForce_.addHostData(make_double3(0.0, 0.0, 0.0));
            bendingTorque_.addHostData(make_double3(0.0, 0.0, 0.0));
            normalForce_.addHostData(0.0);
            torsionTorque_.addHostData(0.0);
            isBonded_.addHostData(1);
        }

        download(stream);
    }

    // ---------- host-side operations ----------
    void removeHost(size_t index)
    {
        objectPointed_.removeHostData(index);
        objectPointing_.removeHostData(index);
        normalForce_.removeHostData(index);
        torsionTorque_.removeHostData(index);
        shearForce_.removeHostData(index);
        bendingTorque_.removeHostData(index);
        contactNormal_.removeHostData(index);
        isBonded_.removeHostData(index);
    }

    void clearHost()
    {
        objectPointed_.clearHostData();
        objectPointing_.clearHostData();
        normalForce_.clearHostData();
        torsionTorque_.clearHostData();
        shearForce_.clearHostData();
        bendingTorque_.clearHostData();
        contactNormal_.clearHostData();
        isBonded_.clearHostData();
    }

    // ---------- host -> device ----------
    void download(cudaStream_t stream)
    {
        objectPointed_.download(stream);
        objectPointing_.download(stream);
        normalForce_.download(stream);
        torsionTorque_.download(stream);
        shearForce_.download(stream);
        bendingTorque_.download(stream);
        contactNormal_.download(stream);
        isBonded_.download(stream);
    }

    // ---------- device -> host ----------
    void upload(cudaStream_t stream)
    {
        objectPointed_.upload(stream);
        objectPointing_.upload(stream);
        normalForce_.upload(stream);
        torsionTorque_.upload(stream);
        shearForce_.upload(stream);
        bendingTorque_.upload(stream);
        contactNormal_.upload(stream);
        isBonded_.upload(stream);
    }

    // ---------- device pointers ----------
    int*      objectPointed()   { return objectPointed_.d_ptr; }
    int*      objectPointing()  { return objectPointing_.d_ptr; }
    double*   normalForce()     { return normalForce_.d_ptr; }
    double*   torsionTorque()   { return torsionTorque_.d_ptr; }
    double3*  shearForce()      { return shearForce_.d_ptr; }
    double3*  bendingTorque()   { return bendingTorque_.d_ptr; }
    double3*  contactNormal()   { return contactNormal_.d_ptr; }
    int*      isBonded()        { return isBonded_.d_ptr; }

    // ---------- host vectors (copies) ----------
    std::vector<int>     objectPointedVector()   { return objectPointed_.getHostData(); }
    std::vector<int>     objectPointingVector()  { return objectPointing_.getHostData(); }
    std::vector<double>  normalForceVector()     { return normalForce_.getHostData(); }
    std::vector<double>  torsionTorqueVector()   { return torsionTorque_.getHostData(); }
    std::vector<double3> shearForceVector()      { return shearForce_.getHostData(); }
    std::vector<double3> bendingTorqueVector()   { return bendingTorque_.getHostData(); }
    std::vector<double3> contactNormalVector()   { return contactNormal_.getHostData(); }
    std::vector<int>     isBondedVector()        { return isBonded_.getHostData(); }
};

struct HertzianContactParameterTable
{
    double* effectiveYoungsModulus{nullptr};                 // E*
    double* effectiveShearModulus{nullptr};                  // G*
    double* restitutionCoefficient{nullptr};                 // e
    double* rollingStiffnessToShearStiffnessRatio{nullptr};  // k_r / k_s
    double* torsionStiffnessToShearStiffnessRatio{nullptr};  // k_t / k_s
    double* slidingFrictionCoefficient{nullptr};             // μ_s
    double* rollingFrictionCoefficient{nullptr};             // μ_r
    double* torsionFrictionCoefficient{nullptr};             // μ_t
};

struct LinearContactParameterTable
{
    double* normalStiffness{nullptr};        // k_n
    double* shearStiffness{nullptr};         // k_s
    double* rollingStiffness{nullptr};       // k_r
    double* torsionStiffness{nullptr};       // k_t

    double* normalDampingCoefficient{nullptr};   // d_n
    double* shearDampingCoefficient{nullptr};    // d_s
    double* rollingDampingCoefficient{nullptr};  // d_r
    double* torsionDampingCoefficient{nullptr};  // d_t

    double* slidingFrictionCoefficient{nullptr}; // μ_s
    double* rollingFrictionCoefficient{nullptr}; // μ_r
    double* torsionFrictionCoefficient{nullptr}; // μ_t
};

struct BondedContactParameterTable
{
    double* bondRadiusMultiplier{nullptr};         // γ: bond radius = γ * min(r)
    double* bondYoungsModulus{nullptr};            // E_bond
    double* normalToShearStiffnessRatio{nullptr};  // k_n / k_s
    double* tensileStrength{nullptr};              // σ_s
    double* cohesion{nullptr};                     // C
    double* frictionCoefficient{nullptr};          // μ
};

struct HertzianRow
{
    int materialIndexA;
    int materialIndexB;

    double effectiveYoungsModulus;                 // E*
    double effectiveShearModulus;                  // G*
    double restitutionCoefficient;                 // e
    double rollingStiffnessToShearStiffnessRatio;  // k_r / k_s
    double torsionStiffnessToShearStiffnessRatio;  // k_t / k_s
    double slidingFrictionCoefficient;             // μ_s
    double rollingFrictionCoefficient;             // μ_r
    double torsionFrictionCoefficient;             // μ_t
};

struct LinearRow
{
    int materialIndexA;
    int materialIndexB;

    double normalStiffness;            // k_n
    double shearStiffness;             // k_s
    double rollingStiffness;           // k_r
    double torsionStiffness;           // k_t

    double normalDampingCoefficient;   // d_n
    double shearDampingCoefficient;    // d_s
    double rollingDampingCoefficient;  // d_r
    double torsionDampingCoefficient;  // d_t

    double slidingFrictionCoefficient; // μ_s
    double rollingFrictionCoefficient; // μ_r
    double torsionFrictionCoefficient; // μ_t
};

struct BondedRow
{
    int materialIndexA;
    int materialIndexB;

    double bondRadiusMultiplier;       // γ
    double bondYoungsModulus;          // E_bond
    double normalToShearStiffnessRatio;// k_n / k_s
    double tensileStrength;            // σ_s
    double cohesion;                   // C
    double frictionCoefficient;        // μ
};

struct contactModelParameters
{
    std::size_t numberOfMaterials{0};
    std::size_t pairTableSize{0};

private:
    // Hertzian
    DeviceArray1D<double> hertzianEffectiveYoungsModulus_;
    DeviceArray1D<double> hertzianEffectiveShearModulus_;
    DeviceArray1D<double> hertzianRestitutionCoefficient_;
    DeviceArray1D<double> hertzianRollingStiffnessToShearStiffnessRatio_;
    DeviceArray1D<double> hertzianTorsionStiffnessToShearStiffnessRatio_;
    DeviceArray1D<double> hertzianSlidingFrictionCoefficient_;
    DeviceArray1D<double> hertzianRollingFrictionCoefficient_;
    DeviceArray1D<double> hertzianTorsionFrictionCoefficient_;

    // Linear
    DeviceArray1D<double> linearNormalStiffness_;
    DeviceArray1D<double> linearShearStiffness_;
    DeviceArray1D<double> linearRollingStiffness_;
    DeviceArray1D<double> linearTorsionStiffness_;
    DeviceArray1D<double> linearNormalDampingCoefficient_;
    DeviceArray1D<double> linearShearDampingCoefficient_;
    DeviceArray1D<double> linearRollingDampingCoefficient_;
    DeviceArray1D<double> linearTorsionDampingCoefficient_;
    DeviceArray1D<double> linearSlidingFrictionCoefficient_;
    DeviceArray1D<double> linearRollingFrictionCoefficient_;
    DeviceArray1D<double> linearTorsionFrictionCoefficient_;

    // Bonded
    DeviceArray1D<double> bondedBondRadiusMultiplier_;
    DeviceArray1D<double> bondedYoungsModulus_;
    DeviceArray1D<double> bondedNormalToShearStiffnessRatio_;
    DeviceArray1D<double> bondedTensileStrength_;
    DeviceArray1D<double> bondedCohesion_;
    DeviceArray1D<double> bondedFrictionCoefficient_;

public:
    HertzianContactParameterTable hertzian;
    LinearContactParameterTable   linear;
    BondedContactParameterTable   bonded;

    contactModelParameters()  = default;
    ~contactModelParameters() = default;
    contactModelParameters(const contactModelParameters&)            = delete;
    contactModelParameters& operator=(const contactModelParameters&) = delete;
    contactModelParameters(contactModelParameters&&) noexcept        = default;
    contactModelParameters& operator=(contactModelParameters&&) noexcept = default;

    void releaseDeviceArray()
    {
        hertzianEffectiveYoungsModulus_.releaseDeviceArray();
        hertzianEffectiveShearModulus_.releaseDeviceArray();
        hertzianRestitutionCoefficient_.releaseDeviceArray();
        hertzianRollingStiffnessToShearStiffnessRatio_.releaseDeviceArray();
        hertzianTorsionStiffnessToShearStiffnessRatio_.releaseDeviceArray();
        hertzianSlidingFrictionCoefficient_.releaseDeviceArray();
        hertzianRollingFrictionCoefficient_.releaseDeviceArray();
        hertzianTorsionFrictionCoefficient_.releaseDeviceArray();

        linearNormalStiffness_.releaseDeviceArray();
        linearShearStiffness_.releaseDeviceArray();
        linearRollingStiffness_.releaseDeviceArray();
        linearTorsionStiffness_.releaseDeviceArray();
        linearNormalDampingCoefficient_.releaseDeviceArray();
        linearShearDampingCoefficient_.releaseDeviceArray();
        linearRollingDampingCoefficient_.releaseDeviceArray();
        linearTorsionDampingCoefficient_.releaseDeviceArray();
        linearSlidingFrictionCoefficient_.releaseDeviceArray();
        linearRollingFrictionCoefficient_.releaseDeviceArray();
        linearTorsionFrictionCoefficient_.releaseDeviceArray();

        bondedBondRadiusMultiplier_.releaseDeviceArray();
        bondedYoungsModulus_.releaseDeviceArray();
        bondedNormalToShearStiffnessRatio_.releaseDeviceArray();
        bondedTensileStrength_.releaseDeviceArray();
        bondedCohesion_.releaseDeviceArray();
        bondedFrictionCoefficient_.releaseDeviceArray();

        numberOfMaterials = 0;
        pairTableSize     = 0;
    }

    std::size_t hostContactParameterArrayIndex(
    int materialIndexA,
    int materialIndexB,
    std::size_t numberOfMaterials,
    std::size_t cap)
    {
        const std::size_t N = numberOfMaterials;
        if (N == 0 || cap == 0) return 0;

        if (materialIndexA < 0 || materialIndexB < 0 ||
            materialIndexA >= static_cast<int>(N) ||
            materialIndexB >= static_cast<int>(N))
        {
            return cap - 1;
        }

        int i = materialIndexA;
        int j = materialIndexB;
        if (i > j) std::swap(i, j);

        std::size_t si = static_cast<std::size_t>(i);
        std::size_t sj = static_cast<std::size_t>(j);

        std::size_t idx = (si * (2 * N - si + 1)) / 2 + (sj - si);
        if (idx >= cap) idx = cap - 1;
        return idx;
    }

    void buildFromTables(const std::vector<HertzianRow>& hertzianTable,
                         const std::vector<LinearRow>&   linearTable,
                         const std::vector<BondedRow>&   bondedTable,
                         cudaStream_t                    stream)
    {
        releaseDeviceArray();

        int maxIndex = -1;
        auto updateMax = [&](int a, int b)
        {
            if (a > maxIndex) maxIndex = a;
            if (b > maxIndex) maxIndex = b;
        };

        for (const auto& row : hertzianTable)
            updateMax(row.materialIndexA, row.materialIndexB);
        for (const auto& row : linearTable)
            updateMax(row.materialIndexA, row.materialIndexB);
        for (const auto& row : bondedTable)
            updateMax(row.materialIndexA, row.materialIndexB);

        if (maxIndex < 0) return;

        numberOfMaterials = static_cast<std::size_t>(maxIndex + 1);
        pairTableSize     = (numberOfMaterials * (numberOfMaterials + 1)) / 2 + 1;

        hertzianEffectiveYoungsModulus_.allocDeviceArray(pairTableSize, stream);
        hertzianEffectiveShearModulus_.allocDeviceArray(pairTableSize, stream);
        hertzianRestitutionCoefficient_.allocDeviceArray(pairTableSize, stream);
        hertzianRollingStiffnessToShearStiffnessRatio_.allocDeviceArray(pairTableSize, stream);
        hertzianTorsionStiffnessToShearStiffnessRatio_.allocDeviceArray(pairTableSize, stream);
        hertzianSlidingFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);
        hertzianRollingFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);
        hertzianTorsionFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);

        linearNormalStiffness_.allocDeviceArray(pairTableSize, stream);
        linearShearStiffness_.allocDeviceArray(pairTableSize, stream);
        linearRollingStiffness_.allocDeviceArray(pairTableSize, stream);
        linearTorsionStiffness_.allocDeviceArray(pairTableSize, stream);
        linearNormalDampingCoefficient_.allocDeviceArray(pairTableSize, stream);
        linearShearDampingCoefficient_.allocDeviceArray(pairTableSize, stream);
        linearRollingDampingCoefficient_.allocDeviceArray(pairTableSize, stream);
        linearTorsionDampingCoefficient_.allocDeviceArray(pairTableSize, stream);
        linearSlidingFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);
        linearRollingFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);
        linearTorsionFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);

        bondedBondRadiusMultiplier_.allocDeviceArray(pairTableSize, stream);
        bondedYoungsModulus_.allocDeviceArray(pairTableSize, stream);
        bondedNormalToShearStiffnessRatio_.allocDeviceArray(pairTableSize, stream);
        bondedTensileStrength_.allocDeviceArray(pairTableSize, stream);
        bondedCohesion_.allocDeviceArray(pairTableSize, stream);
        bondedFrictionCoefficient_.allocDeviceArray(pairTableSize, stream);

        std::vector<double> h_E_star(pairTableSize, 0.0);
        std::vector<double> h_G_star(pairTableSize, 0.0);
        std::vector<double> h_e(pairTableSize, 1.0);
        std::vector<double> h_krks(pairTableSize, 0.0);
        std::vector<double> h_ktks(pairTableSize, 0.0);
        std::vector<double> h_mu_s_h(pairTableSize, 0.0);
        std::vector<double> h_mu_r_h(pairTableSize, 0.0);
        std::vector<double> h_mu_t_h(pairTableSize, 0.0);

        std::vector<double> h_kn(pairTableSize, 0.0);
        std::vector<double> h_ks(pairTableSize, 0.0);
        std::vector<double> h_kr(pairTableSize, 0.0);
        std::vector<double> h_kt(pairTableSize, 0.0);
        std::vector<double> h_dn(pairTableSize, 0.0);
        std::vector<double> h_ds(pairTableSize, 0.0);
        std::vector<double> h_dr(pairTableSize, 0.0);
        std::vector<double> h_dt(pairTableSize, 0.0);
        std::vector<double> h_mu_s_l(pairTableSize, 0.0);
        std::vector<double> h_mu_r_l(pairTableSize, 0.0);
        std::vector<double> h_mu_t_l(pairTableSize, 0.0);

        std::vector<double> h_gamma(pairTableSize, 1.0);
        std::vector<double> h_Eb(pairTableSize, 0.0);
        std::vector<double> h_knks(pairTableSize, 1.0);
        std::vector<double> h_sigma_s(pairTableSize, 0.0);
        std::vector<double> h_C(pairTableSize, 0.0);
        std::vector<double> h_mu_b(pairTableSize, 0.0);

        for (const auto& row : hertzianTable)
        {
            std::size_t idx = hostContactParameterArrayIndex(
                row.materialIndexA, row.materialIndexB,
                numberOfMaterials, pairTableSize);

            h_E_star[idx]  = row.effectiveYoungsModulus;
            h_G_star[idx]  = row.effectiveShearModulus;
            h_e[idx]       = row.restitutionCoefficient;
            h_krks[idx]    = row.rollingStiffnessToShearStiffnessRatio;
            h_ktks[idx]    = row.torsionStiffnessToShearStiffnessRatio;
            h_mu_s_h[idx]  = row.slidingFrictionCoefficient;
            h_mu_r_h[idx]  = row.rollingFrictionCoefficient;
            h_mu_t_h[idx]  = row.torsionFrictionCoefficient;
        }

        for (const auto& row : linearTable)
        {
            std::size_t idx = hostContactParameterArrayIndex(
                row.materialIndexA, row.materialIndexB,
                numberOfMaterials, pairTableSize);

            h_kn[idx]      = row.normalStiffness;
            h_ks[idx]      = row.shearStiffness;
            h_kr[idx]      = row.rollingStiffness;
            h_kt[idx]      = row.torsionStiffness;
            h_dn[idx]      = row.normalDampingCoefficient;
            h_ds[idx]      = row.shearDampingCoefficient;
            h_dr[idx]      = row.rollingDampingCoefficient;
            h_dt[idx]      = row.torsionDampingCoefficient;
            h_mu_s_l[idx]  = row.slidingFrictionCoefficient;
            h_mu_r_l[idx]  = row.rollingFrictionCoefficient;
            h_mu_t_l[idx]  = row.torsionFrictionCoefficient;
        }

        for (const auto& row : bondedTable)
        {
            std::size_t idx = hostContactParameterArrayIndex(
                row.materialIndexA, row.materialIndexB,
                numberOfMaterials, pairTableSize);

            h_gamma[idx]   = row.bondRadiusMultiplier;
            h_Eb[idx]      = row.bondYoungsModulus;
            h_knks[idx]    = row.normalToShearStiffnessRatio;
            h_sigma_s[idx] = row.tensileStrength;
            h_C[idx]       = row.cohesion;
            h_mu_b[idx]    = row.frictionCoefficient;
        }

        cuda_copy(hertzianEffectiveYoungsModulus_.d_ptr,
                  h_E_star.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianEffectiveShearModulus_.d_ptr,
                  h_G_star.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianRestitutionCoefficient_.d_ptr,
                  h_e.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianRollingStiffnessToShearStiffnessRatio_.d_ptr,
                  h_krks.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianTorsionStiffnessToShearStiffnessRatio_.d_ptr,
                  h_ktks.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianSlidingFrictionCoefficient_.d_ptr,
                  h_mu_s_h.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianRollingFrictionCoefficient_.d_ptr,
                  h_mu_r_h.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(hertzianTorsionFrictionCoefficient_.d_ptr,
                  h_mu_t_h.data(), pairTableSize, CopyDir::H2D, stream);

        cuda_copy(linearNormalStiffness_.d_ptr,
                  h_kn.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearShearStiffness_.d_ptr,
                  h_ks.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearRollingStiffness_.d_ptr,
                  h_kr.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearTorsionStiffness_.d_ptr,
                  h_kt.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearNormalDampingCoefficient_.d_ptr,
                  h_dn.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearShearDampingCoefficient_.d_ptr,
                  h_ds.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearRollingDampingCoefficient_.d_ptr,
                  h_dr.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearTorsionDampingCoefficient_.d_ptr,
                  h_dt.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearSlidingFrictionCoefficient_.d_ptr,
                  h_mu_s_l.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearRollingFrictionCoefficient_.d_ptr,
                  h_mu_r_l.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(linearTorsionFrictionCoefficient_.d_ptr,
                  h_mu_t_l.data(), pairTableSize, CopyDir::H2D, stream);

        cuda_copy(bondedBondRadiusMultiplier_.d_ptr,
                  h_gamma.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(bondedYoungsModulus_.d_ptr,
                  h_Eb.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(bondedNormalToShearStiffnessRatio_.d_ptr,
                  h_knks.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(bondedTensileStrength_.d_ptr,
                  h_sigma_s.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(bondedCohesion_.d_ptr,
                  h_C.data(), pairTableSize, CopyDir::H2D, stream);
        cuda_copy(bondedFrictionCoefficient_.d_ptr,
                  h_mu_b.data(), pairTableSize, CopyDir::H2D, stream);

        hertzian.effectiveYoungsModulus                 = hertzianEffectiveYoungsModulus_.d_ptr;
        hertzian.effectiveShearModulus                  = hertzianEffectiveShearModulus_.d_ptr;
        hertzian.restitutionCoefficient                 = hertzianRestitutionCoefficient_.d_ptr;
        hertzian.rollingStiffnessToShearStiffnessRatio  = hertzianRollingStiffnessToShearStiffnessRatio_.d_ptr;
        hertzian.torsionStiffnessToShearStiffnessRatio  = hertzianTorsionStiffnessToShearStiffnessRatio_.d_ptr;
        hertzian.slidingFrictionCoefficient             = hertzianSlidingFrictionCoefficient_.d_ptr;
        hertzian.rollingFrictionCoefficient             = hertzianRollingFrictionCoefficient_.d_ptr;
        hertzian.torsionFrictionCoefficient             = hertzianTorsionFrictionCoefficient_.d_ptr;

        linear.normalStiffness          = linearNormalStiffness_.d_ptr;
        linear.shearStiffness           = linearShearStiffness_.d_ptr;
        linear.rollingStiffness         = linearRollingStiffness_.d_ptr;
        linear.torsionStiffness         = linearTorsionStiffness_.d_ptr;
        linear.normalDampingCoefficient = linearNormalDampingCoefficient_.d_ptr;
        linear.shearDampingCoefficient  = linearShearDampingCoefficient_.d_ptr;
        linear.rollingDampingCoefficient= linearRollingDampingCoefficient_.d_ptr;
        linear.torsionDampingCoefficient= linearTorsionDampingCoefficient_.d_ptr;
        linear.slidingFrictionCoefficient = linearSlidingFrictionCoefficient_.d_ptr;
        linear.rollingFrictionCoefficient = linearRollingFrictionCoefficient_.d_ptr;
        linear.torsionFrictionCoefficient = linearTorsionFrictionCoefficient_.d_ptr;

        bonded.bondRadiusMultiplier        = bondedBondRadiusMultiplier_.d_ptr;
        bonded.bondYoungsModulus           = bondedYoungsModulus_.d_ptr;
        bonded.normalToShearStiffnessRatio = bondedNormalToShearStiffnessRatio_.d_ptr;
        bonded.tensileStrength             = bondedTensileStrength_.d_ptr;
        bonded.cohesion                    = bondedCohesion_.d_ptr;
        bonded.frictionCoefficient         = bondedFrictionCoefficient_.d_ptr;
    }
};

struct interactionMap
{
private:
    DeviceArray1D<int> hashIndex_;
    DeviceArray1D<int> hashValue_;
    DeviceArray1D<int> hashAux_;
    DeviceArray1D<int> countA_;
    DeviceArray1D<int> prefixSumA_;
    DeviceArray1D<int> startB_;
    DeviceArray1D<int> endB_;

    size_t ASize_;
    size_t BSize_;
    size_t hashSize_;
public:
    interactionMap()  = default;
    ~interactionMap() = default;

    interactionMap(const interactionMap&)            = delete;
    interactionMap& operator=(const interactionMap&) = delete;

    interactionMap(interactionMap&&) noexcept            = default;
    interactionMap& operator=(interactionMap&&) noexcept = default;

    void alloc(size_t objectASize, size_t objectBSize, cudaStream_t stream)
    {
        if(objectASize <= 0 || objectBSize <= 0) return;
        ASize_ = objectASize;
        BSize_ = objectBSize;
        countA_.allocDeviceArray(ASize_, stream);
        prefixSumA_.allocDeviceArray(ASize_, stream);
        startB_.allocDeviceArray(BSize_, stream);
        endB_.allocDeviceArray(BSize_, stream);
    }

    void hashInit(int* objectPointing, size_t hashSize, cudaStream_t stream)
    {
        hashSize_ = hashSize;
        if(hashSize > hashValue_.deviceSize()) 
        {
            hashIndex_.allocDeviceArray(hashSize, stream);
            hashValue_.allocDeviceArray(hashSize, stream);
            hashAux_.allocDeviceArray(hashSize, stream);
        }
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr,   0xFF, hashValue_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr,   0xFF, hashIndex_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashAux_.d_ptr,   0xFF, hashAux_.deviceSize() * sizeof(int), stream));
        cuda_copy(hashValue_.d_ptr, objectPointing, hashSize,CopyDir::D2D, stream);

        CUDA_CHECK(cudaMemsetAsync(startB_.d_ptr,   0xFF, startB_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(endB_.d_ptr,   0xFF, endB_.deviceSize() * sizeof(int), stream));
    }

    size_t ASize() const      { return ASize_; }
    size_t BSize() const      { return BSize_; }
    size_t hashSize() const   { return hashSize_; }
    int* hashIndex()          { return hashIndex_.d_ptr; }
    int* hashValue()          { return hashValue_.d_ptr; }
    int* hashAux()            { return hashAux_.d_ptr; }
    int* countA()             { return countA_.d_ptr; }
    int* prefixSumA()         { return prefixSumA_.d_ptr; }
    int* startB()  { return startB_.d_ptr; }
    int* endB()    { return endB_.d_ptr; }
};