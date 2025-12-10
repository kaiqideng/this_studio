#pragma once
#include "solidParticleHandler.h"
#include "wallHandler.h"
#include <cstddef>

struct HertzianParamsList
{
    std::vector<int> materialIndexA;
    std::vector<int> materialIndexB;
    std::vector<double> E;          // Effective Young's Modulus (E*)
    std::vector<double> G;          // Effective Shear Modulus (G*)
    std::vector<double> res;        // Coefficient of Restitution
    std::vector<double> k_r_k_s;    // Rolling Stiffness Ratio (k_r / k_s)
    std::vector<double> k_t_k_s;    // Torsion Stiffness Ratio (k_t / k_s)
    std::vector<double> mu_s;       // Sliding Friction Coefficient
    std::vector<double> mu_r;       // Rolling Friction Coefficient
    std::vector<double> mu_t;       // Torsion Friction Coefficient
};

struct LinearParamsList
{
    std::vector<int> materialIndexA;
    std::vector<int> materialIndexB;
    std::vector<double> k_n;        // Normal Stiffness (Spring constant)
    std::vector<double> k_s;        // Shear Stiffness
    std::vector<double> k_r;        // Rolling Stiffness
    std::vector<double> k_t;        // Torsion Stiffness
    std::vector<double> d_n;        // Normal Damping Coefficient
    std::vector<double> d_s;        // Shear Damping Coefficient
    std::vector<double> d_r;        // Rolling Damping Coefficient
    std::vector<double> d_t;        // Torsion Damping Coefficient
    std::vector<double> mu_s;       // Sliding Friction Coefficient
    std::vector<double> mu_r;       // Rolling Friction Coefficient
    std::vector<double> mu_t;       // Torsion Friction Coefficient
};

struct BondedParamsList
{
    std::vector<int> materialIndexA;
    std::vector<int> materialIndexB;
    std::vector<double> gamma;      // Multiplier for calculating the bond radius
    std::vector<double> E;          // bond Young's Modulus
    std::vector<double> k_n_k_s;    // Normal to Shear Stiffness Ratio (k_n / k_s)
    std::vector<double> sigma_s;    // Tensile Strength
    std::vector<double> C;          // Cohesion
    std::vector<double> mu;         // General Friction Coefficient
};

class DEMHandler:
    public solidParticleHandler, public wallHandler
{
public:
    DEMHandler(cudaStream_t s) : solidParticleHandler(s), wallHandler(s)
    {
        dir = "DEMOutput";
        solidContactModelParametersHostArrayChangedFlag = false;
    }

    ~DEMHandler() = default;

    void setDir(const std::string& name) {dir = name + "Output";}

    void setHerzianModel(int materialIndexA, int materialIndexB, 
        double effectiveYoungsModulus, double effectiveShearModulus, double restitution,
        double rollingSlidingStiffnessRatio, double torsionSlidingStiffnessRatio, 
        double slidingFrictionCoefficient, double rollingFrictionCoefficient, double torsionFrictionCoefficient)
    {
        HertzianContactModelParameters.materialIndexA.push_back(materialIndexA);
        HertzianContactModelParameters.materialIndexB.push_back(materialIndexB);
        HertzianContactModelParameters.E.push_back(effectiveYoungsModulus);
        HertzianContactModelParameters.G.push_back(effectiveShearModulus);
        HertzianContactModelParameters.res.push_back(restitution);
        HertzianContactModelParameters.k_r_k_s.push_back(rollingSlidingStiffnessRatio);
        HertzianContactModelParameters.k_t_k_s.push_back(torsionSlidingStiffnessRatio);
        HertzianContactModelParameters.mu_s.push_back(slidingFrictionCoefficient);
        HertzianContactModelParameters.mu_r.push_back(rollingFrictionCoefficient);
        HertzianContactModelParameters.mu_t.push_back(torsionFrictionCoefficient);

        solidContactModelParametersHostArrayChangedFlag = true;
    }

    void setLinearModel(int materialIndexA, int materialIndexB, 
        double normalStiffness, double shearStiffness, double rollingStiffness, double torsionStiffness, 
        double normalDampingCoefficient, double shearDampingCoefficient, double rollingDampingCoefficient, double torsionDampingCoefficient, 
        double slidingFrictionCoefficient, double rollingFrictionCoefficient, double torsionFrictionCoefficient)
    {
        LinearContactModelParameters.materialIndexA.push_back(materialIndexA);
        LinearContactModelParameters.materialIndexB.push_back(materialIndexB);
        LinearContactModelParameters.k_n.push_back(normalStiffness);
        LinearContactModelParameters.k_s.push_back(shearStiffness);
        LinearContactModelParameters.k_r.push_back(rollingStiffness);
        LinearContactModelParameters.k_t.push_back(torsionStiffness);
        LinearContactModelParameters.d_n.push_back(normalDampingCoefficient);
        LinearContactModelParameters.d_s.push_back(shearDampingCoefficient);
        LinearContactModelParameters.d_r.push_back(rollingDampingCoefficient);
        LinearContactModelParameters.d_t.push_back(torsionDampingCoefficient);
        LinearContactModelParameters.mu_s.push_back(slidingFrictionCoefficient);
        LinearContactModelParameters.mu_r.push_back(rollingFrictionCoefficient);
        LinearContactModelParameters.mu_t.push_back(torsionFrictionCoefficient);

        solidContactModelParametersHostArrayChangedFlag = true;
    }

    void setBondedModel(int materialIndexA, int materialIndexB, 
        double bondRadiusMultiplier, double bondYoungsModulus, double normalToShearStiffnessRatio, 
        double tensileStrength, double cohesion, double frictionCoefficient)
    {
        BondedContactModelParameters.materialIndexA.push_back(materialIndexA);
        BondedContactModelParameters.materialIndexB.push_back(materialIndexB);
        BondedContactModelParameters.gamma.push_back(bondRadiusMultiplier);
        BondedContactModelParameters.E.push_back(bondYoungsModulus);
        BondedContactModelParameters.k_n_k_s.push_back(normalToShearStiffnessRatio);
        BondedContactModelParameters.sigma_s.push_back(tensileStrength);
        BondedContactModelParameters.C.push_back(cohesion);
        BondedContactModelParameters.mu.push_back(frictionCoefficient);

        solidContactModelParametersHostArrayChangedFlag = true;
    }

protected:
    void downLoadSolidContactModelParameters()
    {
        if(!solidContactModelParametersHostArrayChangedFlag) return;
        else solidContactModelParametersHostArrayChangedFlag = false;
        
        size_t num0A = 0, num0B = 0, num1A = 0, num1B = 0, num2A = 0, num2B = 0;
        if(HertzianContactModelParameters.materialIndexA.size() > 0)
        {
            num0A = *std::max_element(HertzianContactModelParameters.materialIndexA.begin(), HertzianContactModelParameters.materialIndexA.end());
            num0B = *std::max_element(HertzianContactModelParameters.materialIndexB.begin(), HertzianContactModelParameters.materialIndexB.end());
        }
        if(LinearContactModelParameters.materialIndexA.size() > 0)
        {
            num1A = *std::max_element(LinearContactModelParameters.materialIndexA.begin(), LinearContactModelParameters.materialIndexA.end());
            num1B = *std::max_element(LinearContactModelParameters.materialIndexB.begin(), LinearContactModelParameters.materialIndexB.end());
        }
        if(BondedContactModelParameters.materialIndexA.size() > 0)
        {
            num2A = *std::max_element(BondedContactModelParameters.materialIndexA.begin(), BondedContactModelParameters.materialIndexA.end());
            num2B = *std::max_element(BondedContactModelParameters.materialIndexB.begin(), BondedContactModelParameters.materialIndexB.end());
        }
        size_t num = std::max(num0A,num0B);
        num = std::max(num,num1A);
        num = std::max(num,num1B);
        num = std::max(num,num2A);
        num = std::max(num,num2B);
        num += 1;

        solidContactModelParameters.setNumberOfMaterials(num,stream);

        for(size_t i = 0; i < HertzianContactModelParameters.E.size(); i++)
        {
            solidContactModelParameters.setHertzian(HertzianContactModelParameters.materialIndexA[i], 
            HertzianContactModelParameters.materialIndexB[i], 
            HertzianContactModelParameters.E[i], 
            HertzianContactModelParameters.G[i], 
            HertzianContactModelParameters.res[i], 
            HertzianContactModelParameters.k_r_k_s[i], 
            HertzianContactModelParameters.k_t_k_s[i], 
            HertzianContactModelParameters.mu_s[i], 
            HertzianContactModelParameters.mu_r[i], 
            HertzianContactModelParameters.mu_t[i], 
            stream);
        }

        for(size_t i = 0; i < LinearContactModelParameters.k_n.size(); i++)
        {
            solidContactModelParameters.setLinear(LinearContactModelParameters.materialIndexA[i], 
            LinearContactModelParameters.materialIndexB[i], 
            LinearContactModelParameters.k_n[i], 
            LinearContactModelParameters.k_s[i], 
            LinearContactModelParameters.k_r[i], 
            LinearContactModelParameters.k_t[i], 
            LinearContactModelParameters.d_n[i], 
            LinearContactModelParameters.d_s[i], 
            LinearContactModelParameters.d_r[i], 
            LinearContactModelParameters.d_t[i],
            LinearContactModelParameters.mu_s[i], 
            LinearContactModelParameters.mu_r[i], 
            LinearContactModelParameters.mu_t[i], 
            stream);
        }

        for(size_t i = 0; i < BondedContactModelParameters.E.size(); i++)
        {
            solidContactModelParameters.setBonded(BondedContactModelParameters.materialIndexA[i], 
            BondedContactModelParameters.materialIndexB[i], 
            BondedContactModelParameters.gamma[i], 
            BondedContactModelParameters.E[i], 
            BondedContactModelParameters.k_n_k_s[i], 
            BondedContactModelParameters.sigma_s[i], 
            BondedContactModelParameters.C[i], 
            BondedContactModelParameters.mu[i], 
            stream);
        }
    }

    std::string getDir() {return dir;}

    void outputSolidParticleVTU(const size_t iFrame, const size_t iStep, const double timeStep);

    virtual bool handleHostArray() {return false;};

    void DEMinitialize(const double3 domainOrigin, const double3 domainSize)
    {
        handleHostArray();

        downLoadSolidContactModelParameters();

        downLoadSolidParticlesInteractions();

        downloadSpatialGrids(domainOrigin, domainSize);

        outputSolidParticleVTU(0, 0, 0.);
    }

    void DEMupdate(const double3 domainOrigin, const double3 domainSize, const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        DEMNeighborSearch(maxThreadsPerBlock);

        DEMIntegrateBeforeContact(gravity, timeStep, maxThreadsPerBlock);

        clearParticleForceTorque();

        DEMInteractionCalculation(timeStep, maxThreadsPerBlock);

        if(handleHostArray())
        {
            downLoadSolidContactModelParameters();

            downLoadSolidParticlesInteractions();

            downloadSpatialGrids(domainOrigin, domainSize);
        }

        DEMIntegrateAfterContact(gravity, timeStep, maxThreadsPerBlock);
    }

private:
    void DEMNeighborSearch(const size_t maxThreadsPerBlock)
    {
        solidParticleNeighborSearch(solidParticleInteractions, solidParticles, spatialGrids, maxThreadsPerBlock, stream);
    }

    void DEMIntegrateBeforeContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        solidParticleIntegrateBeforeContact(solidParticles, clumps, gravity, timeStep, maxThreadsPerBlock, stream);
    }

    void DEMInteractionCalculation(const double timeStep, const size_t maxThreadsPerBlock)
    {
        solidParticleInteractionCalculation(solidParticleInteractions, bondedSolidParticleInteractions, solidParticles, clumps, 
        solidContactModelParameters, timeStep, maxThreadsPerBlock, stream);
    }

    void DEMIntegrateAfterContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        solidParticleIntegrateAfterContact(solidParticles, clumps, gravity, timeStep, maxThreadsPerBlock, stream);
    }

    std::string dir;
    bool solidContactModelParametersHostArrayChangedFlag;
    HertzianParamsList HertzianContactModelParameters;
    LinearParamsList LinearContactModelParameters;
    BondedParamsList BondedContactModelParameters;

    solidContactModelParameter solidContactModelParameters;
};