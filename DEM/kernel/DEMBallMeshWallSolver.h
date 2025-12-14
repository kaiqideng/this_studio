#pragma once
#include "DEMBallSolver.h"
#include "wallHandler.h"

class DEMBallMeshWallSolver:
    public DEMBallSolver, public wallHandler
{
public:
    DEMBallMeshWallSolver(cudaStream_t s) : DEMBallSolver(s), wallHandler(s)
	{
        iStep_ = 0;
		iFrame_ = 0;
		time_ = 0.0;
    }

    ~DEMBallMeshWallSolver() = default;

    const size_t &getStep() const override {return iStep_;}

	const size_t &getFrame() const override {return iFrame_;}

	const double &getTime() const override {return time_;} 

    void solve() override 
    {
        removeVtuFiles(getDir());
        removeDatFiles(getDir());
        const double timeStep = getTimeStep();

        if (initialize()) 
        {
            size_t numSteps = size_t((getMaximumTime()) / timeStep) + 1;
            size_t frameInterval = numSteps / getNumFrames();
            if (frameInterval < 1) frameInterval = 1;

            while (iStep_ <= numSteps) 
            {
                iStep_++;
                time_+= timeStep;
                ballNeighborSearch(getGPUThreadsPerBlock());
                ball1stHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
                ballContactCalculation(contactModelParams(), timeStep,getGPUThreadsPerBlock());
                if(handleHostArray())
                {
                    ballInitialize(getDomainOrigin(), getDomainSize());
                    wallInitialize(getDomainOrigin(), getDomainSize());
                    downloadContactModelParameters();
                }
                ball2ndHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
                if (iStep_ % frameInterval == 0) 
                {
                    iFrame_++;
                    std::cout << "DEM solver: frame " << iFrame_ << " at time " << time_ << std::endl;
                    outputBallVTU(getDir(), iFrame_, iStep_, time_);
                    outputBallVTU(getDir(), iFrame_, iStep_, time_);
                }
            }
        }
    }

protected:
    
private:
    bool initialize() override
    {
        std::cout << "DEM solver: initializing..." << std::endl;

        std::cout << "DEM solver: using GPU Device " << getGPUDeviceIndex() << std::endl;
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(getGPUDeviceIndex());
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "DEM solver: cudaSetDevice( " << getGPUDeviceIndex()
                    << " ) failed!  Do you have a CUDA-capable GPU installed?"
                    << std::endl;
            exit(1);
        }
        std::cout << "DEM solver: downloading array from host to device..."
                    << std::endl;
        ballInitialize(getDomainOrigin(), getDomainSize());
        wallInitialize(getDomainOrigin(), getDomainSize());
        ballNeighborSearch(getGPUThreadsPerBlock());
        handleHostArray();
        downloadContactModelParameters();
        outputBallVTU(getDir(), 0, 0, 0.0);
        outputMeshWallVTU(getDir(), 0, 0, 0.0);
        std::cout << "DEM solver: initialization completed." << std::endl;
        return true;
    }

    size_t iStep_;
    size_t iFrame_;;
	double time_;
};