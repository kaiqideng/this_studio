#pragma once
#include "DEMClumpSolver.h"
#include "wallHandler.h"
#include "cudaKernel/ballMeshWallNeighbor.h"
#include "cudaKernel/ballMeshWallIntegration.h"

class DEMClumpMeshWallSolver:
    public DEMClumpSolver
{
public:
    DEMClumpMeshWallSolver(cudaStream_t s) : DEMClumpSolver(s)
	{
        stream_ = s;
    }

    ~DEMClumpMeshWallSolver() = default;

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
            ballTriangleInteractions_.alloc(numBalls1, stream_);
        }

        size_t numTriangles0 = wallHandler_.meshWalls().triangles().deviceSize();
        wallHandler_.download(getDomainOrigin(), getDomainSize(), stream_);
        size_t numTriangles1 = wallHandler_.meshWalls().triangles().deviceSize();
        if(numTriangles1 > numTriangles0)
        {
            ballTriangleInteractionMap_.alloc(numBalls1, numTriangles1, stream_);
        }

        getClumpHandler().download(stream_);
    }

    void outputData() override
	{
		getBallHandler().outputBallVTU(getDir(), getFrame(), getStep(), getTime());
        wallHandler_.outputMeshWallVTU(getDir(), getFrame(), getStep(), getTime());
	}

	void neighborSearch() override
	{
		launchBallNeighborSearch(getBallInteractions(), 
		getBallInteractionMap(), 
		getBallHandler().balls(),
		getBallSpatialGrids(),
		getGPUMaxThreadsPerBlock(),
		stream_);

        launchBallTriangleNeighborSearch(ballTriangleInteractions_, 
        ballTriangleInteractionMap_, 
        getBallHandler().balls(), 
        wallHandler_.meshWalls(), 
        wallHandler_.spatialGrids(), 
        getGPUMaxThreadsPerBlock(), 
        stream_);
	}

	void integration1st(const double dt) override
	{
        launchClump1stHalfIntegration(getClumpHandler().clumps(), 
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

        launchMeshWallIntegration(wallHandler_.meshWalls(), 
        dt, 
        getGPUMaxThreadsPerBlock(), 
        stream_);
	}

	void contactCalculation(const double dt) override
	{
        launchBallMeshWallInteractionCalculation(ballTriangleInteractions_, 
        getBallHandler().balls(), 
        wallHandler_.meshWalls(), 
        getContactModelParams(), 
        ballTriangleInteractionMap_, 
        dt, 
        getGPUMaxThreadsPerBlock(), 
        stream_);

		launchBallContactCalculation(getBallInteractions(), 
		getBondedBallInteractions(), 
		getBallHandler().balls(), 
		getContactModelParams(), 
		getBallInteractionMap(),
		dt, 
		getGPUMaxThreadsPerBlock(), 
		stream_);
	}

    void integration2nd(const double dt) override
	{
        launchClump2ndHalfIntegration(getClumpHandler().clumps(), 
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

    wallHandler wallHandler_;
    solidInteraction ballTriangleInteractions_;
    interactionMap ballTriangleInteractionMap_;
};