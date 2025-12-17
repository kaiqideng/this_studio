#pragma once
#include "solverParams.h"
#include "contactModelParams.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"
#include "ballHandler.h"
#include "cudaKernel/myStruct/interaction.h"
#include "cudaKernel/ballNeighborSearch.h"
#include "cudaKernel/ballIntegration.h"

class DEMBaseSolver:
    public solverParams, public contactModelParams
{
public:
    DEMBaseSolver(cudaStream_t s) : solverParams(), contactModelParams()
	{
		setNameFlag_ = false;
		dir_ = "DEMOutput";
		iStep_ = 0;
		iFrame_ = 0;
		time_ = 0.0;

		stream_ = s;
	}

	~DEMBaseSolver() = default;

	void setProblemName(const std::string& name) 
	{
		if(!setNameFlag_) dir_ = name; 
		setNameFlag_ = true;
	}

	void addCluster(std::vector<double3> positions, 
    std::vector<double3> velocities, 
    std::vector<double3> angularVelocities, 
    std::vector<double> radius, 
    double density, 
    int materialID)
    {
        ballHandler_.addCluster(positions, velocities, angularVelocities, radius, density, materialID, stream_);
    }

    void addFixedCluster(std::vector<double3> positions, 
    std::vector<double> radius, 
    int materialID)
    {
        ballHandler_.addFixedCluster(positions, radius, materialID, stream_);
    }

    void solve();

protected:
    const std::string& getDir() const {return dir_;}

	const size_t& getStep() const {return iStep_;}

    const size_t& getFrame() const {return iFrame_;}

	const double& getTime() const {return time_;}

    virtual bool handleHostArrayInLoop();

	ballHandler& getBallHandler();

	spatialGrid& getBallSpatialGrids();

	solidInteraction& getBallInteractions() {return ballInteractions_;}

	interactionMap& getBallInteractionMap() {return ballInteractionMap_;}

	bondedInteraction& getBondedBallInteractions() {return bondedBallInteractions_;}

	void addBondedObjects(const std::vector<int> &object0, const std::vector<int> &object1);

	void addBallExternalForce(const std::vector<double3>& externalForce);

    void addBallExternalTorque(const std::vector<double3>& externalTorque);

private:
    //Be fucking careful to this motherfucker, making sure that you have correct "A" size and "B" size of the interactionMap
    virtual void download();

    bool initialize();

	virtual void outputData();

	virtual void neighborSearch();

	virtual void integration1st(const double dt);

	virtual void contactCalculation(const double dt);

	virtual void integration2nd(const double dt);

    bool setNameFlag_;
    std::string dir_;
    size_t iStep_;
    size_t iFrame_;;
	double time_;

	cudaStream_t stream_;

	ballHandler ballHandler_;

    solidInteraction ballInteractions_;
    bondedInteraction bondedBallInteractions_;
	interactionMap ballInteractionMap_;
};