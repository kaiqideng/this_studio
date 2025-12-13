#pragma once
#include "myStruct/interaction.h"
#include "solverParams.h"
#include "ballHandler.h"
#include <vector>

class DEMBallSolver:
    public solverParams, public ballHandler
{
public:
    DEMBallSolver(cudaStream_t s) : solverParams(), ballHandler(s)
	{
		stream_ = s;
		setNameFlag_ = false;
		dir_ = "DEMBallSolverOutput";
		numSteps_ = 1;
		iStep_ = 0;
		iFrame_ = 0;
		frameInterval_ = 1;
		time_ = 0.0;

		setModelFlag_ = false;
	}

	~DEMBallSolver() = default;

	void setProblemName(const std::string& name) 
	{
		if(!setNameFlag_) dir_ = name + "Output"; 
		setNameFlag_ = true;
	}

	void setHertzianContactModelForPair(
		int    materialIdA,
		int    materialIdB,
		double effectiveYoungModulus,         // E*
		double effectiveShearModulus,         // G*
		double coefficientOfRestitution,      // e
		double rollingStiffnessRatio_kr_over_ks,
		double torsionStiffnessRatio_kt_over_ks,
		double slidingFrictionCoeff_mu_s,
		double rollingFrictionCoeff_mu_r,
		double torsionFrictionCoeff_mu_t)
	{
		if (materialIdA > materialIdB)
			std::swap(materialIdA, materialIdB);

		hertzianRows_.push_back(HertzianRow{
			materialIdA,
			materialIdB,
			effectiveYoungModulus,            // -> E
			effectiveShearModulus,            // -> G
			coefficientOfRestitution,         // -> res
			rollingStiffnessRatio_kr_over_ks, // -> k_r_k_s
			torsionStiffnessRatio_kt_over_ks, // -> k_t_k_s
			slidingFrictionCoeff_mu_s,        // -> mu_s
			rollingFrictionCoeff_mu_r,        // -> mu_r
			torsionFrictionCoeff_mu_t         // -> mu_t
		});

		setModelFlag_ = true;
	};

	void setLinearContactModelForPair(
		int    materialIdA,
		int    materialIdB,
		double normalStiffness_k_n,
		double shearStiffness_k_s,
		double rollingStiffness_k_r,
		double torsionStiffness_k_t,
		double normalDamping_d_n,
		double shearDamping_d_s,
		double rollingDamping_d_r,
		double torsionDamping_d_t,
		double slidingFrictionCoeff_mu_s,
		double rollingFrictionCoeff_mu_r,
		double torsionFrictionCoeff_mu_t)
	{
		if (materialIdA > materialIdB)
			std::swap(materialIdA, materialIdB);

		linearRows_.push_back(LinearRow{
			materialIdA,
			materialIdB,
			normalStiffness_k_n,              // -> k_n
			shearStiffness_k_s,               // -> k_s
			rollingStiffness_k_r,             // -> k_r
			torsionStiffness_k_t,             // -> k_t
			normalDamping_d_n,                // -> d_n
			shearDamping_d_s,                 // -> d_s
			rollingDamping_d_r,               // -> d_r
			torsionDamping_d_t,               // -> d_t
			slidingFrictionCoeff_mu_s,        // -> mu_s
			rollingFrictionCoeff_mu_r,        // -> mu_r
			torsionFrictionCoeff_mu_t         // -> mu_t
		});

		setModelFlag_ = true;
	};

	void setBondedContactModelForPair(
		int    materialIdA,
		int    materialIdB,
		double bondRadiusMultiplier_gamma,
		double bondYoungModulus_Eb,
		double normalToShearStiffnessRatio_k_n_over_k_s,
		double tensileStrength_sigma_s,
		double cohesion_C,
		double frictionCoeff_mu)
	{
		if (materialIdA > materialIdB)
			std::swap(materialIdA, materialIdB);

		bondedRows_.push_back(BondedRow{
			materialIdA,
			materialIdB,
			bondRadiusMultiplier_gamma,           // -> gamma
			bondYoungModulus_Eb,                  // -> E
			normalToShearStiffnessRatio_k_n_over_k_s, // -> k_n_k_s
			tensileStrength_sigma_s,              // -> sigma_s
			cohesion_C,                           // -> C
			frictionCoeff_mu                      // -> mu
		});

		setModelFlag_ = true;
	};

    virtual void solve();

protected:
	double getStep()const {return iStep_;}

	double getTime() {return time_;} 

	contactModelParameters &contactModelParams() {return contactModelParams_;}

private:
    bool initialize();

	void downloadContactModelParameters()
	{
		if(setModelFlag_)
		{
			contactModelParams_.buildFromTables(hertzianRows_, 
			linearRows_, 
			bondedRows_, 
			stream_);
			setModelFlag_ = false;
		}
	}

	cudaStream_t stream_;
    bool setNameFlag_;
    std::string dir_;
    size_t numSteps_;
    size_t iStep_;
    size_t iFrame_;;
    size_t frameInterval_;
	double time_;

	std::vector<HertzianRow> hertzianRows_;
	std::vector<LinearRow> linearRows_;
	std::vector<BondedRow> bondedRows_;
    
	bool setModelFlag_ = false;
	contactModelParameters contactModelParams_;
};