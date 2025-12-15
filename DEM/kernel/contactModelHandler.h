#pragma once
#include "myStruct/interaction.h"

class contactModelHandler
{
public:
    contactModelHandler(cudaStream_t s)
    {
        stream_ = s;
        downloadFlag_ = false;
    }

    ~contactModelHandler() = default;

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

		downloadFlag_ = true;
	};

	void setLinearContactModelForPair(
		int    materialIdA,
		int    materialIdB,
		double normalStiffness_k_n,
		double slidingStiffness_k_s,
		double rollingStiffness_k_r,
		double torsionStiffness_k_t,
		double normalDamping_d_n,
		double slidingDamping_d_s,
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
			slidingStiffness_k_s,               // -> k_s
			rollingStiffness_k_r,             // -> k_r
			torsionStiffness_k_t,             // -> k_t
			normalDamping_d_n,                // -> d_n
			slidingDamping_d_s,                 // -> d_s
			rollingDamping_d_r,               // -> d_r
			torsionDamping_d_t,               // -> d_t
			slidingFrictionCoeff_mu_s,        // -> mu_s
			rollingFrictionCoeff_mu_r,        // -> mu_r
			torsionFrictionCoeff_mu_t         // -> mu_t
		});

		downloadFlag_ = true;
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

		downloadFlag_ = true;
	};

protected:
    contactModelParameters &contactModelParams() {return contactModelParams_;}

    void downloadContactModelParameters()
	{
		if(downloadFlag_)
		{
			contactModelParams_.buildFromTables(hertzianRows_, 
			linearRows_, 
			bondedRows_, 
			stream_);
			downloadFlag_ = false;
		}
	}
    
private:
    cudaStream_t stream_;
    bool downloadFlag_;

    std::vector<HertzianRow> hertzianRows_;
	std::vector<LinearRow> linearRows_;
	std::vector<BondedRow> bondedRows_;
    
	contactModelParameters contactModelParams_;
};