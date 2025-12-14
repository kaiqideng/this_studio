#pragma once
#include "myStruct/myUtility/myFileEdit.h"
#include "solverParams.h"
#include "ballHandler.h"
#include "contactModelHandler.h"

class DEMBallSolver:
    public solverParams, public ballHandler, public contactModelHandler
{
public:
    DEMBallSolver(cudaStream_t s) : solverParams(), ballHandler(s), contactModelHandler(s)
	{
		stream_ = s;
		setNameFlag_ = false;

		dir_ = "DEMBallSolverOutput";
		
		iStep_ = 0;
		iFrame_ = 0;
		time_ = 0.0;
	}

	~DEMBallSolver() = default;

	virtual void setProblemName(const std::string& name) 
	{
		if(!setNameFlag_) dir_ = name + "Output"; 
		setNameFlag_ = true;
	}

	const std::string &getDir() const {return dir_;}

	virtual const size_t &getStep() const {return iStep_;}

	virtual const size_t &getFrame() const {return iFrame_;}

	virtual const double &getTime() const {return time_;} 

    virtual bool handleHostArray(){return false;}

    virtual void solve();

private:
    virtual bool initialize();

	cudaStream_t stream_;
    bool setNameFlag_;
	
    std::string dir_;

    size_t iStep_;
    size_t iFrame_;;
	double time_;
};