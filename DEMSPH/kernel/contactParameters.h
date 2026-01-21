#pragma once
#include "myUtility/myHostDeviceArray.h"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cuda_runtime.h>

// You must provide CUDA_CHECK macro yourself.

// ============================================================================
// Input rows (readable inputs)
// ============================================================================
struct HertzianRow
{
    int materialIndexA;
    int materialIndexB;

    double effectiveYoungsModulus;                 // E*
    double effectiveShearModulus;                  // G*
    double restitutionCoefficient;                 // e
    double rollingStiffnessToShearStiffnessRatio;  // k_r / k_s
    double torsionStiffnessToShearStiffnessRatio;  // k_t / k_s
    double slidingFrictionCoefficient;             // mu_s
    double rollingFrictionCoefficient;             // mu_r
    double torsionFrictionCoefficient;             // mu_t
};

struct LinearRow
{
    int materialIndexA;
    int materialIndexB;

    double normalStiffness;            // k_n
    double slidingStiffness;           // k_s
    double rollingStiffness;           // k_r
    double torsionStiffness;           // k_t

    double normalDampingCoefficient;   // d_n
    double slidingDampingCoefficient;  // d_s
    double rollingDampingCoefficient;  // d_r
    double torsionDampingCoefficient;  // d_t

    double slidingFrictionCoefficient; // mu_s
    double rollingFrictionCoefficient; // mu_r
    double torsionFrictionCoefficient; // mu_t
};

struct BondedRow
{
    int materialIndexA;
    int materialIndexB;

    double bondRadiusMultiplier;        // gamma
    double bondYoungsModulus;           // E_bond
    double normalToShearStiffnessRatio; // k_n / k_s
    double tensileStrength;             // sigma_s
    double cohesion;                    // C
    double frictionCoefficient;         // mu
};

// ============================================================================
// Packed parameter layout: param-major
//   ptr[param * cap + pairIdx]
// ============================================================================
enum HertzianParam : int
{
    H_E_STAR = 0,
    H_G_STAR,
    H_RES,
    H_KRKS,
    H_KTKS,
    H_MU_S,
    H_MU_R,
    H_MU_T,
    H_COUNT
};

enum LinearParam : int
{
    L_KN = 0,
    L_KS,
    L_KR,
    L_KT,
    L_DN,
    L_DS,
    L_DR,
    L_DT,
    L_MU_S,
    L_MU_R,
    L_MU_T,
    L_COUNT
};

enum BondedParam : int
{
    B_GAMMA = 0,
    B_EB,
    B_KNKS,
    B_SIGMA_S,
    B_C,
    B_MU,
    B_COUNT
};

// ============================================================================
// Constant memory "background" parameters for kernels
//   NOTE:
//   - Constant memory stores ONLY pointers + meta.
//   - Real parameter arrays live in global memory.
// ============================================================================
struct ContactParamsDevice
{
    int nMaterials {0};
    int cap {0}; // pairTableSize

    const double* hertzian {nullptr}; // size = H_COUNT * cap
    const double* linear   {nullptr}; // size = L_COUNT * cap
    const double* bonded   {nullptr}; // size = B_COUNT * cap
};

extern __constant__ ContactParamsDevice contactPara;

// ============================================================================
// Pair index (host/device)
//   - symmetric (i,j)==(j,i)
//   - last slot (cap-1) is fallback for invalid indices
// ============================================================================
__host__ __device__ inline int contactPairParameterIndex(int a, int b, int nMaterials, int cap)
{
    if (nMaterials <= 0 || cap <= 0) return 0;

    if (a < 0 || b < 0 || a >= nMaterials || b >= nMaterials)
    {
        return cap - 1;
    }

    int i = a, j = b;
    if (i > j) { int t = i; i = j; j = t; }

    long long idx = (static_cast<long long>(i) * (2LL * nMaterials - i + 1LL)) / 2LL
                    + static_cast<long long>(j - i);

    if (idx < 0) idx = 0;
    if (idx >= cap) idx = cap - 1;
    return static_cast<int>(idx);
}

// ============================================================================
// Device access helpers
// ============================================================================
__device__ inline double getHertzianParam(const int pairIdx, const int p)
{
    return contactPara.hertzian[p * contactPara.cap + pairIdx];
}

__device__ inline double getLinearParam(const int pairIdx, const int p)
{
    return contactPara.linear[p * contactPara.cap + pairIdx];
}

__device__ inline double getBondedParam(const int pairIdx, const int p)
{
    return contactPara.bonded[p * contactPara.cap + pairIdx];
}

// ============================================================================
// contactModelParameters (host-side owner + constant-memory commit)
// ============================================================================
class contactModelParameters
{
public:
    std::size_t numberOfMaterials {0};
    std::size_t pairTableSize {0};

private:
    // ---------------------------------------------------------------------
    // Packed arrays (param-major)
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double> hertzianPacked_; // H_COUNT * pairTableSize
    HostDeviceArray1D<double> linearPacked_;   // L_COUNT * pairTableSize
    HostDeviceArray1D<double> bondedPacked_;   // B_COUNT * pairTableSize

private:
    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------
    static int inferNumberOfMaterials_(const std::vector<HertzianRow>& hertzianTable,
                                      const std::vector<LinearRow>& linearTable,
                                      const std::vector<BondedRow>& bondedTable)
    {
        int maxIndex = -1;

        auto updateMax = [&](int a, int b)
        {
            maxIndex = std::max(maxIndex, a);
            maxIndex = std::max(maxIndex, b);
        };

        for (const auto& row : hertzianTable) updateMax(row.materialIndexA, row.materialIndexB);
        for (const auto& row : linearTable)   updateMax(row.materialIndexA, row.materialIndexB);
        for (const auto& row : bondedTable)   updateMax(row.materialIndexA, row.materialIndexB);

        return (maxIndex < 0) ? 0 : (maxIndex + 1);
    }

    static std::size_t computePairTableSize_(int nMaterials)
    {
        if (nMaterials <= 0) return 0;
        return (static_cast<std::size_t>(nMaterials) * (static_cast<std::size_t>(nMaterials) + 1)) / 2 + 1;
    }

    static void setPacked_(std::vector<double>& buf,
                           const std::size_t cap,
                           const int param,
                           const std::size_t idx,
                           const double value)
    {
        buf[static_cast<std::size_t>(param) * cap + idx] = value;
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    contactModelParameters() = default;
    ~contactModelParameters() = default;

    contactModelParameters(const contactModelParameters&) = delete;
    contactModelParameters& operator=(const contactModelParameters&) = delete;

    contactModelParameters(contactModelParameters&&) noexcept = default;
    contactModelParameters& operator=(contactModelParameters&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Build + upload + commit constant memory
    // ---------------------------------------------------------------------
    void buildFromTables(const std::vector<HertzianRow>& hertzianTable,
                         const std::vector<LinearRow>&   linearTable,
                         const std::vector<BondedRow>&   bondedTable,
                         cudaStream_t stream)
    {
        // --------------------------
        // Infer sizes
        // --------------------------
        const int nMat = inferNumberOfMaterials_(hertzianTable, linearTable, bondedTable);
        if (nMat <= 0) return;

        numberOfMaterials = static_cast<std::size_t>(nMat);
        pairTableSize = computePairTableSize_(nMat);
        if (pairTableSize == 0) return;

        // --------------------------
        // Host buffers with defaults
        // --------------------------
        std::vector<double> H(static_cast<std::size_t>(H_COUNT) * pairTableSize, 0.0);
        std::vector<double> L(static_cast<std::size_t>(L_COUNT) * pairTableSize, 0.0);
        std::vector<double> B(static_cast<std::size_t>(B_COUNT) * pairTableSize, 0.0);

        // Default: Hertzian e = 1
        for (std::size_t idx = 0; idx < pairTableSize; ++idx)
        {
            setPacked_(H, pairTableSize, H_RES, idx, 1.0);
        }

        // Default: Bonded gamma = 1, kn/ks = 1
        for (std::size_t idx = 0; idx < pairTableSize; ++idx)
        {
            setPacked_(B, pairTableSize, B_GAMMA, idx, 1.0);
            setPacked_(B, pairTableSize, B_KNKS,  idx, 1.0);
        }

        auto pairIdx = [&](int a, int b) -> std::size_t
        {
            return static_cast<std::size_t>(contactPairParameterIndex(a, b, nMat, static_cast<int>(pairTableSize)));
        };

        // --------------------------
        // Fill from input tables
        // --------------------------
        for (const auto& row : hertzianTable)
        {
            const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

            setPacked_(H, pairTableSize, H_E_STAR, idx, row.effectiveYoungsModulus);
            setPacked_(H, pairTableSize, H_G_STAR, idx, row.effectiveShearModulus);
            setPacked_(H, pairTableSize, H_RES,    idx, row.restitutionCoefficient);
            setPacked_(H, pairTableSize, H_KRKS,   idx, row.rollingStiffnessToShearStiffnessRatio);
            setPacked_(H, pairTableSize, H_KTKS,   idx, row.torsionStiffnessToShearStiffnessRatio);
            setPacked_(H, pairTableSize, H_MU_S,   idx, row.slidingFrictionCoefficient);
            setPacked_(H, pairTableSize, H_MU_R,   idx, row.rollingFrictionCoefficient);
            setPacked_(H, pairTableSize, H_MU_T,   idx, row.torsionFrictionCoefficient);
        }

        for (const auto& row : linearTable)
        {
            const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

            setPacked_(L, pairTableSize, L_KN,   idx, row.normalStiffness);
            setPacked_(L, pairTableSize, L_KS,   idx, row.slidingStiffness);
            setPacked_(L, pairTableSize, L_KR,   idx, row.rollingStiffness);
            setPacked_(L, pairTableSize, L_KT,   idx, row.torsionStiffness);

            setPacked_(L, pairTableSize, L_DN,   idx, row.normalDampingCoefficient);
            setPacked_(L, pairTableSize, L_DS,   idx, row.slidingDampingCoefficient);
            setPacked_(L, pairTableSize, L_DR,   idx, row.rollingDampingCoefficient);
            setPacked_(L, pairTableSize, L_DT,   idx, row.torsionDampingCoefficient);

            setPacked_(L, pairTableSize, L_MU_S, idx, row.slidingFrictionCoefficient);
            setPacked_(L, pairTableSize, L_MU_R, idx, row.rollingFrictionCoefficient);
            setPacked_(L, pairTableSize, L_MU_T, idx, row.torsionFrictionCoefficient);
        }

        for (const auto& row : bondedTable)
        {
            const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

            setPacked_(B, pairTableSize, B_GAMMA,   idx, row.bondRadiusMultiplier);
            setPacked_(B, pairTableSize, B_EB,      idx, row.bondYoungsModulus);
            setPacked_(B, pairTableSize, B_KNKS,    idx, row.normalToShearStiffnessRatio);
            setPacked_(B, pairTableSize, B_SIGMA_S, idx, row.tensileStrength);
            setPacked_(B, pairTableSize, B_C,       idx, row.cohesion);
            setPacked_(B, pairTableSize, B_MU,      idx, row.frictionCoefficient);
        }

        // --------------------------
        // Upload packed arrays
        // --------------------------
        hertzianPacked_.setHost(H);
        linearPacked_.setHost(L);
        bondedPacked_.setHost(B);

        hertzianPacked_.copyHostToDevice(stream);
        linearPacked_.copyHostToDevice(stream);
        bondedPacked_.copyHostToDevice(stream);

        // --------------------------
        // Commit constant memory (pointers + meta)
        // --------------------------
        ContactParamsDevice dev;
        dev.nMaterials = static_cast<int>(numberOfMaterials);
        dev.cap = static_cast<int>(pairTableSize);
        dev.hertzian = hertzianPacked_.d_ptr;
        dev.linear = linearPacked_.d_ptr;
        dev.bonded = bondedPacked_.d_ptr;

        CUDA_CHECK(cudaMemcpyToSymbolAsync(contactPara,
                                           &dev,
                                           sizeof(ContactParamsDevice),
                                           0,
                                           cudaMemcpyHostToDevice,
                                           stream));
    }
};