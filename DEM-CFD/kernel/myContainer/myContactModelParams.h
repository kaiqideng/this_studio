#pragma once
#include "myHash.h"
#include <vector>

/**
 * @brief Inner struct containing parameters for the Hertzian contact model.
 * Does NOT manage its own memory.
 */
struct HertzianParams
{
    double* E{ nullptr };          // Effective Young's Modulus (E*)
    double* G{ nullptr };          // Effective Shear Modulus (G*)
    double* res{ nullptr };        // Coefficient of Restitution
    double* k_r_k_s{ nullptr };    // Rolling Stiffness Ratio (k_r / k_s)
    double* k_t_k_s{ nullptr };    // Torsion Stiffness Ratio (k_t / k_s)
    double* mu_s{ nullptr };       // Static Friction Coefficient (Sliding)
    double* mu_r{ nullptr };       // Rolling Friction Coefficient
    double* mu_t{ nullptr };       // Torsion Friction Coefficient
};

/**
 * @brief Inner struct containing parameters for the Linear contact model.
 * Does NOT manage its own memory.
 */
struct LinearParams
{
    double* k_n{ nullptr };        // Normal Stiffness (Spring constant)
    double* k_s{ nullptr };        // Shear Stiffness
    double* k_r{ nullptr };        // Rolling Stiffness
    double* k_t{ nullptr };        // Torsion Stiffness
    double* d_n{ nullptr };        // Normal Damping Coefficient
    double* d_s{ nullptr };        // Shear Damping Coefficient
    double* d_r{ nullptr };        // Rolling Damping Coefficient
    double* d_t{ nullptr };        // Torsion Damping Coefficient
    double* mu_s{ nullptr };       // Static Friction Coefficient (Sliding)
    double* mu_r{ nullptr };       // Rolling Friction Coefficient
    double* mu_t{ nullptr };       // Torsion Friction Coefficient
};

/**
 * @brief Inner struct containing parameters for the Bonded contact model.
 * Does NOT manage its own memory.
 */
struct BondedParams
{
    double* gamma{ nullptr };      // Multiplier for calculating the bond radius
    double* E{ nullptr };          // bond Young's Modulus
    double* k_n_k_s{ nullptr };    // Normal to Shear Stiffness Ratio (k_n / k_s)
    double* sigma_s{ nullptr };    // Tensile Strength
    double* C{ nullptr };          // Cohesion
    double* mu{ nullptr };         // General Friction Coefficient
};

struct solidContactModelParameter
{
private:
    size_t numberOfMaterials{ 0 };
    size_t d_size{ 0 };

    void release()
    {
        if (hertzian.E) { CUDA_FREE(hertzian.E); hertzian.E = nullptr; }
        if (hertzian.G) { CUDA_FREE(hertzian.G); hertzian.G = nullptr; }
        if (hertzian.res) { CUDA_FREE(hertzian.res); hertzian.res = nullptr; }
        if (hertzian.k_r_k_s) { CUDA_FREE(hertzian.k_r_k_s); hertzian.k_r_k_s = nullptr; }
        if (hertzian.k_t_k_s) { CUDA_FREE(hertzian.k_t_k_s); hertzian.k_t_k_s = nullptr; }
        if (hertzian.mu_s) { CUDA_FREE(hertzian.mu_s); hertzian.mu_s = nullptr; }
        if (hertzian.mu_r) { CUDA_FREE(hertzian.mu_r); hertzian.mu_r = nullptr; }
        if (hertzian.mu_t) { CUDA_FREE(hertzian.mu_t); hertzian.mu_t = nullptr; }

        if (linear.k_n) { CUDA_FREE(linear.k_n); linear.k_n = nullptr; }
        if (linear.k_s) { CUDA_FREE(linear.k_s); linear.k_s = nullptr; }
        if (linear.k_r) { CUDA_FREE(linear.k_r); linear.k_r = nullptr; }
        if (linear.k_t) { CUDA_FREE(linear.k_t); linear.k_t = nullptr; }
        if (linear.d_n) { CUDA_FREE(linear.d_n); linear.d_n = nullptr; }
        if (linear.d_s) { CUDA_FREE(linear.d_s); linear.d_s = nullptr; }
        if (linear.d_r) { CUDA_FREE(linear.d_r); linear.d_r = nullptr; }
        if (linear.d_t) { CUDA_FREE(linear.d_t); linear.d_t = nullptr; }
        if (linear.mu_s) { CUDA_FREE(linear.mu_s); linear.mu_s = nullptr; }
        if (linear.mu_r) { CUDA_FREE(linear.mu_r); linear.mu_r = nullptr; }
        if (linear.mu_t) { CUDA_FREE(linear.mu_t); linear.mu_t = nullptr; }

        if (bonded.gamma) { CUDA_FREE(bonded.gamma); bonded.gamma = nullptr; }
        if (bonded.E) { CUDA_FREE(bonded.E); bonded.E = nullptr; }
        if (bonded.k_n_k_s) { CUDA_FREE(bonded.k_n_k_s); bonded.k_n_k_s = nullptr; }
        if (bonded.sigma_s) { CUDA_FREE(bonded.sigma_s); bonded.sigma_s = nullptr; }
        if (bonded.C) { CUDA_FREE(bonded.C); bonded.C = nullptr; }
        if (bonded.mu) { CUDA_FREE(bonded.mu); bonded.mu = nullptr; }
    }

    void alloc(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) release(); 

        d_size = n;

        CUDA_ALLOC(hertzian.E, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.G, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.res, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.k_r_k_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.k_t_k_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.mu_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.mu_r, n, InitMode::ZERO, stream);
        CUDA_ALLOC(hertzian.mu_t, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.k_n, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.k_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.k_r, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.k_t, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.d_n, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.d_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.d_r, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.d_t, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.mu_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.mu_r, n, InitMode::ZERO, stream);
        CUDA_ALLOC(linear.mu_t, n, InitMode::ZERO, stream);
        CUDA_ALLOC(bonded.gamma, n, InitMode::ZERO, stream);
        CUDA_ALLOC(bonded.E, n, InitMode::ZERO, stream);
        CUDA_ALLOC(bonded.k_n_k_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(bonded.sigma_s, n, InitMode::ZERO, stream);
        CUDA_ALLOC(bonded.C, n, InitMode::ZERO, stream);
        CUDA_ALLOC(bonded.mu, n, InitMode::ZERO, stream);

        std::vector<double> iniValue(n,1.0);
        cuda_copy(hertzian.res,iniValue.data(),n,CopyDir::H2D,stream);
        cuda_copy(bonded.gamma,iniValue.data(),n,CopyDir::H2D,stream);
        cuda_copy(bonded.k_n_k_s,iniValue.data(),n,CopyDir::H2D,stream);
    }

public:
    HertzianParams hertzian;
    LinearParams linear;
    BondedParams bonded;

    solidContactModelParameter() = default;
    ~solidContactModelParameter() { release(); } 
    solidContactModelParameter(const solidContactModelParameter&) = delete;

    solidContactModelParameter(solidContactModelParameter&& other) noexcept 
    { *this = std::move(other); }

    solidContactModelParameter& operator=(solidContactModelParameter&& other) noexcept
    {
        if (this != &other) 
        {
            release();

            numberOfMaterials = std::exchange(other.numberOfMaterials, 0);
            d_size= std::exchange(other.d_size, 0);
            
            hertzian.E = std::exchange(other.hertzian.E, nullptr);
            hertzian.G = std::exchange(other.hertzian.G, nullptr);
            hertzian.res = std::exchange(other.hertzian.res, nullptr);
            hertzian.k_r_k_s = std::exchange(other.hertzian.k_r_k_s, nullptr);
            hertzian.k_t_k_s = std::exchange(other.hertzian.k_t_k_s, nullptr);
            hertzian.mu_s = std::exchange(other.hertzian.mu_s, nullptr);
            hertzian.mu_r = std::exchange(other.hertzian.mu_r, nullptr);
            hertzian.mu_t = std::exchange(other.hertzian.mu_t, nullptr);

            linear.k_n = std::exchange(other.linear.k_n, nullptr);
            linear.k_s = std::exchange(other.linear.k_s, nullptr);
            linear.k_r = std::exchange(other.linear.k_r, nullptr);
            linear.k_t = std::exchange(other.linear.k_t, nullptr);
            linear.d_n = std::exchange(other.linear.d_n, nullptr);
            linear.d_s = std::exchange(other.linear.d_s, nullptr);
            linear.d_r = std::exchange(other.linear.d_r, nullptr);
            linear.d_t = std::exchange(other.linear.d_t, nullptr);
            linear.mu_s = std::exchange(other.linear.mu_s, nullptr);
            linear.mu_r = std::exchange(other.linear.mu_r, nullptr);
            linear.mu_t = std::exchange(other.linear.mu_t, nullptr);

            bonded.gamma = std::exchange(other.bonded.gamma, nullptr);
            bonded.E = std::exchange(other.bonded.E, nullptr);
            bonded.k_n_k_s = std::exchange(other.bonded.k_n_k_s, nullptr);
            bonded.sigma_s = std::exchange(other.bonded.sigma_s, nullptr);
            bonded.C = std::exchange(other.bonded.C, nullptr);
            bonded.mu = std::exchange(other.bonded.mu, nullptr);
        }
        return *this;
    }

    void setNumberOfMaterials(size_t numMaterials, cudaStream_t stream)
    {
        numberOfMaterials = numMaterials;
        
        alloc(numMaterials * (numMaterials + 1) / 2, stream);
    }

    size_t getArraryIndex(const int materialIndexA, const int materialIndexB)
    {
        size_t N = numberOfMaterials;

        if (N == 0 || d_size == 0) return 0;

        if (materialIndexA >= static_cast<int>(N) || materialIndexB >= static_cast<int>(N) || materialIndexA < 0 || materialIndexB < 0) 
        {
            return d_size - 1; 
        }
    
        size_t i = static_cast<size_t>(materialIndexA);
        size_t j = static_cast<size_t>(materialIndexB);
        if (i > j) 
        { 
            size_t t = i; i = j; j = t;
        }

        return (i * (2 * N - i + 1)) / 2 + j - i;
    }

    void setHertzian(int mA, int mB, 
        double E, double G, double res, double k_r_k_s, 
        double k_t_k_s, double mu_s, double mu_r, double mu_t,
        cudaStream_t stream)
    {
        size_t index = getArraryIndex(mA, mB);
        if(index >= d_size) return;

        cuda_copy(hertzian.E + index, &E, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.G + index, &G, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.res + index, &res, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.k_r_k_s + index, &k_r_k_s, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.k_t_k_s + index, &k_t_k_s, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.mu_s + index, &mu_s, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.mu_r + index, &mu_r, 1, CopyDir::H2D, stream);
        cuda_copy(hertzian.mu_t + index, &mu_t, 1, CopyDir::H2D, stream);
    }

    void setLinear(int mA, int mB, 
        double k_n, double k_s, double k_r, double k_t, 
        double d_n, double d_s, double d_r, double d_t, 
        double mu_s, double mu_r, double mu_t,
        cudaStream_t stream)
    {
        size_t index = getArraryIndex(mA, mB);
        if(index >= d_size) return; 

        cuda_copy(linear.k_n + index, &k_n, 1, CopyDir::H2D, stream);
        cuda_copy(linear.k_s + index, &k_s, 1, CopyDir::H2D, stream);
        cuda_copy(linear.k_r + index, &k_r, 1, CopyDir::H2D, stream);
        cuda_copy(linear.k_t + index, &k_t, 1, CopyDir::H2D, stream);

        cuda_copy(linear.d_n + index, &d_n, 1, CopyDir::H2D, stream);
        cuda_copy(linear.d_s + index, &d_s, 1, CopyDir::H2D, stream);
        cuda_copy(linear.d_r + index, &d_r, 1, CopyDir::H2D, stream);
        cuda_copy(linear.d_t + index, &d_t, 1, CopyDir::H2D, stream);

        cuda_copy(linear.mu_s + index, &mu_s, 1, CopyDir::H2D, stream);
        cuda_copy(linear.mu_r + index, &mu_r, 1, CopyDir::H2D, stream);
        cuda_copy(linear.mu_t + index, &mu_t, 1, CopyDir::H2D, stream);
    }

    void setBonded(int mA, int mB, 
        double gamma, double E, double k_n_k_s, double sigma_s, 
        double C, double mu,
        cudaStream_t stream)
    {
        size_t index = getArraryIndex(mA, mB);
        if(index >= d_size) return; 

        cuda_copy(bonded.gamma + index, &gamma, 1, CopyDir::H2D, stream);
        cuda_copy(bonded.E + index, &E, 1, CopyDir::H2D, stream);
        cuda_copy(bonded.k_n_k_s + index, &k_n_k_s, 1, CopyDir::H2D, stream);
        cuda_copy(bonded.sigma_s + index, &sigma_s, 1, CopyDir::H2D, stream);
        cuda_copy(bonded.C + index, &C, 1, CopyDir::H2D, stream);
        cuda_copy(bonded.mu + index, &mu, 1, CopyDir::H2D, stream);
    }

    size_t getNumberOfMaterials() const {return numberOfMaterials;}

    size_t size() const {return d_size;}
};