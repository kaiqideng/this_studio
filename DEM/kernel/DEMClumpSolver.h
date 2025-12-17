#pragma once
#include "DEMBaseSolver.h"
#include "clumpHandler.h"

class DEMClumpSolver:
    public DEMBaseSolver
{
public:
    DEMClumpSolver(cudaStream_t s) : DEMBaseSolver(s)
	{
        stream_ = s;
    }

    ~DEMClumpSolver() = default;

    void addClump(std::vector<double3> points, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    double3 velocity, 
    double3 angularVelocity, 
    double mass, 
    symMatrix inertiaTensor, 
    int materialID)
    {
        clumpHandler_.addClump(points, 
        radius, 
        centroidPosition, 
        velocity, 
        angularVelocity, 
        mass, 
        inertiaTensor, 
        materialID, 
        getBallHandler(), 
        stream_);
    }

    void addFixedClump(std::vector<double3> points, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    int materialID)
    {
        clumpHandler_.addFixedClump(points, 
        radius, 
        centroidPosition, 
        materialID, 
        getBallHandler(), 
        stream_);
    }

protected:
    clumpHandler& getClumpHandler() {return clumpHandler_;}
    
private:
    void download() override
    {
        downloadContactModelParams(stream_);
        
        size_t numBalls0 = getBallHandler().balls().deviceSize();
        getBallHandler().download(getDomainOrigin(), getDomainSize(), stream_);
        size_t numBalls1 = getBallHandler().balls().deviceSize();
        if(numBalls1 > numBalls0)
        {
            getBallInteractions().alloc(numBalls1 * 6, stream_);
            getBallInteractionMap().alloc(numBalls1, numBalls1, stream_);
        }

        clumpHandler_.download(stream_);
    }

    void integration1st(const double dt) override
	{
        launchClump1stHalfIntegration(clumpHandler_.clumps(), 
        getBallHandler().balls(), 
        getGravity(), 
        dt, 
        getGPUMaxThreadsPerBlock(), 
        stream_);

		launchBall1stHalfIntegration(getBallHandler().balls(), 
		getGravity(), 
		dt, 
		getGPUMaxThreadsPerBlock(), 
		stream_);
	}

    void integration2nd(const double dt) override
	{
        launchClump2ndHalfIntegration(clumpHandler_.clumps(), 
        getBallHandler().balls(), 
        getGravity(), 
        dt, 
        getGPUMaxThreadsPerBlock(), 
        stream_);

		launchBall2ndHalfIntegration(getBallHandler().balls(), 
		getGravity(), 
		dt, 
		getGPUMaxThreadsPerBlock(), 
		stream_);
	}

    cudaStream_t stream_;

    clumpHandler clumpHandler_;
};