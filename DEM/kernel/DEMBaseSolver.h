#pragma once
#include "solverParams.h"
#include "contactModelParams.h"
#include "ballHandler.h"

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

private:
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
};