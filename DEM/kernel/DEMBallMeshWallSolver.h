#pragma once
#include "DEMBallSolver.h"
#include "myStruct/interaction.h"
#include "wallHandler.h"
#include "ballMeshWallNeighbor.h"
#include "ballMeshWallIntegration.h"

class DEMBallMeshWallSolver:
    public DEMBallSolver, public wallHandler
{
public:
    DEMBallMeshWallSolver(cudaStream_t s) : DEMBallSolver(s), wallHandler(s)
	{
        stream_ = s;
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
            if(handleHostArrayInLoop())
            {
                initializeDeviceSide();
            }
            size_t numSteps = size_t((getMaximumTime()) / timeStep) + 1;
            size_t frameInterval = numSteps / getNumFrames();
            if (frameInterval < 1) frameInterval = 1;

            while (iStep_ <= numSteps) 
            {
                iStep_++;
                time_+= timeStep;
                ballNeighborSearch(getGPUThreadsPerBlock());
                ballMeshWallNeighborSearch(getGPUThreadsPerBlock());
                meshWall1stHalfIntegration(getGravity(), timeStep, getGPUThreadsPerBlock());
                ball1stHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
                ballMeshWallContactCalculation(timeStep, getGPUThreadsPerBlock());
                ballContactCalculation(contactModelParams(), timeStep,getGPUThreadsPerBlock());
                if(handleHostArrayInLoop())
                {
                    initializeDeviceSide();
                }
                ball2ndHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
                meshWall2ndHalfIntegration(getGravity(), timeStep, getGPUThreadsPerBlock());
                if (iStep_ % frameInterval == 0) 
                {
                    iFrame_++;
                    std::cout << "DEM solver: frame " << iFrame_ << " at time " << time_ << std::endl;
                    outputBallVTU(getDir(), iFrame_, iStep_, time_);
                    outputMeshWallVTU(getDir(), iFrame_, iStep_, time_);
                }
            }
        }
    }

protected:
    void ballMeshWallNeighborSearch(const size_t maxThreadsPerBlock)
    {
        launchBallTriangleNeighborSearch(ballTriangleInteractions_, 
        ballTriangleInteractionMap_, 
        balls(), 
        meshWalls(), 
        meshWallSpatialGrids(), 
        maxThreadsPerBlock, 
        stream_);
    }

    void meshWall1stHalfIntegration(const double3 g, const double dt, const size_t maxThreadsPerBlock)
    {
        launchMeshWall1stHalfIntegration(meshWalls(), 
        dt, 
        maxThreadsPerBlock, 
        stream_);
    }

    void ballMeshWallContactCalculation(const double dt, const size_t maxThreadsPerBlock)
    {
        launchBallMeshWallInteractionCalculation(ballTriangleInteractions_, 
        balls(), 
        meshWalls(), 
        contactModelParams(), 
        ballTriangleInteractionMap_, 
        dt, 
        maxThreadsPerBlock, 
        stream_);
    }

    void meshWall2ndHalfIntegration(const double3 g, const double dt, const size_t maxThreadsPerBlock)
    {
        launchMeshWall2ndHalfIntegration(meshWalls(), 
        dt, 
        maxThreadsPerBlock, 
        stream_);
    }
    
private:
    void initializeDeviceSide() override
    {
        ballInitialize(getDomainOrigin(), getDomainSize());
        wallInitialize(getDomainOrigin(), getDomainSize());
        ballTriangleInteractions_.alloc(balls().deviceSize(), stream_);
        ballTriangleInteractionMap_.alloc(balls().deviceSize(), meshWalls().triangles().deviceSize(), stream_);
        downloadContactModelParameters();
    }

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
        initializeDeviceSide();
        if(balls().deviceSize() == 0)
        {
            std::cout << "DEM solver: initialization failed" << std::endl;
            return false;
        } 
        ballNeighborSearch(getGPUThreadsPerBlock());
        ballMeshWallNeighborSearch(getGPUThreadsPerBlock());
        outputBallVTU(getDir(), 0, 0, 0.0);
        outputMeshWallVTU(getDir(), 0, 0, 0.0);
        std::cout << "DEM solver: initialization completed." << std::endl;
        return true;
    }

    cudaStream_t stream_;
    size_t iStep_;
    size_t iFrame_;;
	double time_;

    solidInteraction ballTriangleInteractions_;
    interactionMap ballTriangleInteractionMap_;
};