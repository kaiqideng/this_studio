#pragma once
#include "myContainer/myHash.h"
#include "myContainer/myInteraction.h"
#include "myContainer/myParticle.h"
#include "myContainer/myWall.h"
#include "solidParticleHandler.h"
#include "wallHandler.h"

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
        DEMStream_ = s;
        solidContactModelParametersHostArrayChangedFlag_ = false;
    }

    ~DEMHandler() = default;

    void setHerzianModel(int materialIndexA, int materialIndexB, 
        double effectiveYoungsModulus, double effectiveShearModulus, double restitution,
        double rollingSlidingStiffnessRatio, double torsionSlidingStiffnessRatio, 
        double slidingFrictionCoefficient, double rollingFrictionCoefficient, double torsionFrictionCoefficient)
    {
        HertzianContactModelParameters_.materialIndexA.push_back(materialIndexA);
        HertzianContactModelParameters_.materialIndexB.push_back(materialIndexB);
        HertzianContactModelParameters_.E.push_back(effectiveYoungsModulus);
        HertzianContactModelParameters_.G.push_back(effectiveShearModulus);
        HertzianContactModelParameters_.res.push_back(restitution);
        HertzianContactModelParameters_.k_r_k_s.push_back(rollingSlidingStiffnessRatio);
        HertzianContactModelParameters_.k_t_k_s.push_back(torsionSlidingStiffnessRatio);
        HertzianContactModelParameters_.mu_s.push_back(slidingFrictionCoefficient);
        HertzianContactModelParameters_.mu_r.push_back(rollingFrictionCoefficient);
        HertzianContactModelParameters_.mu_t.push_back(torsionFrictionCoefficient);

        solidContactModelParametersHostArrayChangedFlag_ = true;
    }

    void setLinearModel(int materialIndexA, int materialIndexB, 
        double normalStiffness, double shearStiffness, double rollingStiffness, double torsionStiffness, 
        double normalDampingCoefficient, double shearDampingCoefficient, double rollingDampingCoefficient, double torsionDampingCoefficient, 
        double slidingFrictionCoefficient, double rollingFrictionCoefficient, double torsionFrictionCoefficient)
    {
        LinearContactModelParameters_.materialIndexA.push_back(materialIndexA);
        LinearContactModelParameters_.materialIndexB.push_back(materialIndexB);
        LinearContactModelParameters_.k_n.push_back(normalStiffness);
        LinearContactModelParameters_.k_s.push_back(shearStiffness);
        LinearContactModelParameters_.k_r.push_back(rollingStiffness);
        LinearContactModelParameters_.k_t.push_back(torsionStiffness);
        LinearContactModelParameters_.d_n.push_back(normalDampingCoefficient);
        LinearContactModelParameters_.d_s.push_back(shearDampingCoefficient);
        LinearContactModelParameters_.d_r.push_back(rollingDampingCoefficient);
        LinearContactModelParameters_.d_t.push_back(torsionDampingCoefficient);
        LinearContactModelParameters_.mu_s.push_back(slidingFrictionCoefficient);
        LinearContactModelParameters_.mu_r.push_back(rollingFrictionCoefficient);
        LinearContactModelParameters_.mu_t.push_back(torsionFrictionCoefficient);

        solidContactModelParametersHostArrayChangedFlag_ = true;
    }

    void setBondedModel(int materialIndexA, int materialIndexB, 
        double bondRadiusMultiplier, double bondYoungsModulus, double normalToShearStiffnessRatio, 
        double tensileStrength, double cohesion, double frictionCoefficient)
    {
        BondedContactModelParameters_.materialIndexA.push_back(materialIndexA);
        BondedContactModelParameters_.materialIndexB.push_back(materialIndexB);
        BondedContactModelParameters_.gamma.push_back(bondRadiusMultiplier);
        BondedContactModelParameters_.E.push_back(bondYoungsModulus);
        BondedContactModelParameters_.k_n_k_s.push_back(normalToShearStiffnessRatio);
        BondedContactModelParameters_.sigma_s.push_back(tensileStrength);
        BondedContactModelParameters_.C.push_back(cohesion);
        BondedContactModelParameters_.mu.push_back(frictionCoefficient);

        solidContactModelParametersHostArrayChangedFlag_ = true;
    }

protected:
    virtual bool handleDEMHostArray() {return false;}

    void DEMInitialize(const double3 domainOrigin, const double3 domainSize, const size_t maxThreadsPerBlock)
    {
        downLoadSolidContactModelParameters();

        solidParticleInitialize(domainOrigin, domainSize);

        wallInitialize();

        if(infiniteWalls().deviceSize() > 0) 
        {
            solidParticleInfiniteWallInteractions_.current.allocDeviceArray(solidParticles().hostSize(), DEMStream_);
            solidParticleInfiniteWallNeighbor_.alloc(solidParticles().deviceSize(), DEMStream_);
            infiniteWallInteractionRange_.alloc(infiniteWalls().deviceSize(), DEMStream_);
        }

        solidParticleNeighborSearch(maxThreadsPerBlock);

        solidParticleInfiniteWallNeighborSearch(maxThreadsPerBlock);

        handleDEMHostArray();
    }

    void DEMUpdate(const double3 domainOrigin, const double3 domainSize, const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        solidParticleNeighborSearch(maxThreadsPerBlock);

        solidParticleInfiniteWallNeighborSearch(maxThreadsPerBlock);

        solidParticleIntegrateBeforeContact(gravity, timeStep, maxThreadsPerBlock);

        wallIntegrateBeforeContact(gravity, timeStep, maxThreadsPerBlock);

        solidParticleInteractionCalculation(solidContactModelParameters_, timeStep, maxThreadsPerBlock);

        solidParticleInfiniteWallInteractionCalculation(timeStep, maxThreadsPerBlock);

        if(handleDEMHostArray())
        {
            downLoadSolidContactModelParameters();

            solidParticleInitialize(domainOrigin, domainSize);

            wallInitialize();

            if(infiniteWalls().deviceSize() > 0) 
            {
                solidParticleInfiniteWallInteractions_.current.allocDeviceArray(solidParticles().hostSize(), DEMStream_);
                solidParticleInfiniteWallNeighbor_.alloc(solidParticles().deviceSize(), DEMStream_);
                infiniteWallInteractionRange_.alloc(infiniteWalls().deviceSize(), DEMStream_);
            }
        }

        solidParticleIntegrateAfterContact(gravity, timeStep, maxThreadsPerBlock);

        wallIntegrateAfterContact(gravity, timeStep, maxThreadsPerBlock);
    }

private:
    void downLoadSolidContactModelParameters()
    {
        if(!solidContactModelParametersHostArrayChangedFlag_) return;
        else solidContactModelParametersHostArrayChangedFlag_ = false;
        
        size_t num0A = 0, num0B = 0, num1A = 0, num1B = 0, num2A = 0, num2B = 0;
        if(HertzianContactModelParameters_.materialIndexA.size() > 0)
        {
            num0A = *std::max_element(HertzianContactModelParameters_.materialIndexA.begin(), HertzianContactModelParameters_.materialIndexA.end());
            num0B = *std::max_element(HertzianContactModelParameters_.materialIndexB.begin(), HertzianContactModelParameters_.materialIndexB.end());
        }
        if(LinearContactModelParameters_.materialIndexA.size() > 0)
        {
            num1A = *std::max_element(LinearContactModelParameters_.materialIndexA.begin(), LinearContactModelParameters_.materialIndexA.end());
            num1B = *std::max_element(LinearContactModelParameters_.materialIndexB.begin(), LinearContactModelParameters_.materialIndexB.end());
        }
        if(BondedContactModelParameters_.materialIndexA.size() > 0)
        {
            num2A = *std::max_element(BondedContactModelParameters_.materialIndexA.begin(), BondedContactModelParameters_.materialIndexA.end());
            num2B = *std::max_element(BondedContactModelParameters_.materialIndexB.begin(), BondedContactModelParameters_.materialIndexB.end());
        }
        size_t num = std::max(num0A,num0B);
        num = std::max(num,num1A);
        num = std::max(num,num1B);
        num = std::max(num,num2A);
        num = std::max(num,num2B);
        num += 1;

        solidContactModelParameters_.setNumberOfMaterials(num,DEMStream_);

        for(size_t i = 0; i < HertzianContactModelParameters_.E.size(); i++)
        {
            solidContactModelParameters_.setHertzian(HertzianContactModelParameters_.materialIndexA[i], 
            HertzianContactModelParameters_.materialIndexB[i], 
            HertzianContactModelParameters_.E[i], 
            HertzianContactModelParameters_.G[i], 
            HertzianContactModelParameters_.res[i], 
            HertzianContactModelParameters_.k_r_k_s[i], 
            HertzianContactModelParameters_.k_t_k_s[i], 
            HertzianContactModelParameters_.mu_s[i], 
            HertzianContactModelParameters_.mu_r[i], 
            HertzianContactModelParameters_.mu_t[i], 
            DEMStream_);
        }

        for(size_t i = 0; i < LinearContactModelParameters_.k_n.size(); i++)
        {
            solidContactModelParameters_.setLinear(LinearContactModelParameters_.materialIndexA[i], 
            LinearContactModelParameters_.materialIndexB[i], 
            LinearContactModelParameters_.k_n[i], 
            LinearContactModelParameters_.k_s[i], 
            LinearContactModelParameters_.k_r[i], 
            LinearContactModelParameters_.k_t[i], 
            LinearContactModelParameters_.d_n[i], 
            LinearContactModelParameters_.d_s[i], 
            LinearContactModelParameters_.d_r[i], 
            LinearContactModelParameters_.d_t[i],
            LinearContactModelParameters_.mu_s[i], 
            LinearContactModelParameters_.mu_r[i], 
            LinearContactModelParameters_.mu_t[i], 
            DEMStream_);
        }

        for(size_t i = 0; i < BondedContactModelParameters_.E.size(); i++)
        {
            solidContactModelParameters_.setBonded(BondedContactModelParameters_.materialIndexA[i], 
            BondedContactModelParameters_.materialIndexB[i], 
            BondedContactModelParameters_.gamma[i], 
            BondedContactModelParameters_.E[i], 
            BondedContactModelParameters_.k_n_k_s[i], 
            BondedContactModelParameters_.sigma_s[i], 
            BondedContactModelParameters_.C[i], 
            BondedContactModelParameters_.mu[i], 
            DEMStream_);
        }
    }

    void solidParticleInfiniteWallNeighborSearch(const size_t maxThreadsPerBlock)
    {
        launchSolidParticleInfiniteWallNeighborSearch(solidParticleInfiniteWallInteractions_, 
        solidParticles(), 
        infiniteWalls(), 
        solidParticleInfiniteWallNeighbor_, 
        infiniteWallInteractionRange_, 
        maxThreadsPerBlock, 
        DEMStream_);
    }

    void solidParticleInfiniteWallInteractionCalculation(const double timeStep, const size_t maxThreadsPerBlock)
    {

    }

    cudaStream_t DEMStream_;
    bool solidContactModelParametersHostArrayChangedFlag_;
    HertzianParamsList HertzianContactModelParameters_;
    LinearParamsList LinearContactModelParameters_;
    BondedParamsList BondedContactModelParameters_;

    solidContactModelParameter solidContactModelParameters_;

    interactionSpringSystem solidParticleInfiniteWallInteractions_;
    objectNeighborPrefix solidParticleInfiniteWallNeighbor_;
    sortedHashValueIndex infiniteWallInteractionRange_;

    interactionSpringSystem solidParticleTriangleWallInteractions_;
    objectNeighborPrefix solidParticleTriangleWallNeighbor_;
    sortedHashValueIndex triangleWallInteractionRange_;
};