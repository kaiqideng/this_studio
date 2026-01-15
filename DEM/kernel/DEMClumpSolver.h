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
        gridDim_ = 1;
        blockDim_ = 1;
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
    clumpHandler& getClumpHandler() { return clumpHandler_; }

    void setClumpGPUGridDim(size_t gridDim) { gridDim_ = gridDim; }

    void setClumpGPUBlockDim(size_t blockDim) { blockDim_ = blockDim;}

    const size_t& getClumpGPUGridDim() const { return gridDim_; }

    const size_t& getClumpGPUBlockDim() const { return blockDim_; }
    
private:
    void upload() override
    {
        uploadContactModelParams(stream_);
        getBallHandler().upload(getDomainOrigin(), getDomainSize(), stream_);
        clumpHandler_.upload(stream_);

        const size_t numBalls = getBallHandler().getBalls().deviceSize();
        const size_t maxThreads = getGPUMaxThreadsPerBlock();
        if (numBalls > 0 && maxThreads > 1) setBallGPUBlockDim(maxThreads < numBalls ? maxThreads : numBalls);
        const size_t ballBlockDim = getBallGPUBlockDim();
        if (ballBlockDim > 0) setBallGPUGridDim((numBalls + ballBlockDim - 1) / ballBlockDim);

        const size_t numClumps = clumpHandler_.getClumps().deviceSize();
        if (numClumps > 0 && maxThreads > 1) blockDim_ = maxThreads < numClumps ? maxThreads : numClumps;
        if (blockDim_ > 0) gridDim_ = (numClumps + blockDim_ - 1) / blockDim_;
    }

    void integration1st(const double dt) override
	{
        getClumpHandler().integration1st(getBallHandler().getBalls(), getGravity(), dt, gridDim_, blockDim_, stream_);
        getBallHandler().integration1st(getGravity(), dt, getBallGPUGridDim(), getBallGPUBlockDim(), stream_);
	}

    void integration2nd(const double dt) override
	{
        getClumpHandler().integration2nd(getBallHandler().getBalls(), getGravity(), dt, gridDim_, blockDim_, stream_);
        getBallHandler().integration2nd(getGravity(), dt, getBallGPUGridDim(), getBallGPUBlockDim(), stream_);
	}

    cudaStream_t stream_;

    clumpHandler clumpHandler_;
    size_t gridDim_;
    size_t blockDim_;
};