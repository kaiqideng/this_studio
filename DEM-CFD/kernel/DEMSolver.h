#pragma once
#include "solverParams.h"
#include "DEMHandler.h"

class DEMSolver:
    public solverParams, public DEMHandler
{
public:
    DEMSolver(cudaStream_t s) : solverParams(), DEMHandler(s)
	{
		numSteps = 1;
		iStep = 0;
		iFrame = 0;
		frameInterval = 1;
		time = 0;
	}

	~DEMSolver() = default;

    void solve();

protected:
	const double getStep()const {return iStep;}

	double getTime() {return time;} 

private:
    bool initialize();

    size_t numSteps;
    size_t iStep;
    size_t iFrame;;
    size_t frameInterval;
	double time;
};