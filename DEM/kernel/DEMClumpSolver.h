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
        getBallHandler().download(getDomainOrigin(), getDomainSize(), stream_);
        clumpHandler_.download(stream_);
    }

    void integration1st(const double dt) override
	{
        getClumpHandler().integration1st(getBallHandler().getBalls(), getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
        getBallHandler().integration1st(getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
	}

    void integration2nd(const double dt) override
	{
        getClumpHandler().integration2nd(getBallHandler().getBalls(), getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
        getBallHandler().integration2nd(getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
	}

    cudaStream_t stream_;

    clumpHandler clumpHandler_;
};