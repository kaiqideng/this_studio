#pragma once
#include "DEMBaseSolver.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "wallHandler.h"
#include "cudaKernel/ballMeshWallNeighbor.h"
#include "cudaKernel/ballMeshWallIntegration.h"

class DEMBallMeshWallSolver:
    public DEMBaseSolver
{
public:
    DEMBallMeshWallSolver(cudaStream_t s) : DEMBaseSolver(s)
	{
        stream_ = s;
    }

    ~DEMBallMeshWallSolver() = default;

    void addTriangleMeshWall(const std::vector<double3> &vertices, 
    const std::vector<int3> &triIndices, 
    const double3 &posistion, 
    const double3 &velocity, 
    const double3 &angularVelocity, 
    int matirialID)
    {
        wallHandler_.addTriangleMeshWall(vertices, 
        triIndices, 
        posistion, 
        velocity, 
        angularVelocity, 
        matirialID, 
        stream_);
    }
    
private:
    void download() override
    {
        downloadContactModelParams(stream_);
        getBallHandler().download(getDomainOrigin(), getDomainSize(), stream_);
        wallHandler_.download(stream_);

        std::vector<double> rad = getBallHandler().getBalls().radiusVector();
        double maxBallDiameter = 0.0;
        if(rad.size() > 0) maxBallDiameter = *std::max_element(rad.begin(), rad.end()) * 2.0;
        double cellSizeOneDim = wallHandler_.getMeshWalls().getMaxEdgeLength() * 1.2;
        cellSizeOneDim = std::max(cellSizeOneDim, maxBallDiameter);
        if(cellSizeOneDim > triangleSpatialGrids_.cellSize.x 
        || cellSizeOneDim > triangleSpatialGrids_.cellSize.y 
        || cellSizeOneDim > triangleSpatialGrids_.cellSize.z)
        {
            triangleSpatialGrids_.set(getDomainOrigin(), getDomainSize(), cellSizeOneDim, stream_);
        }
        ballTriangleInteractions_.alloc(getBallHandler().getBalls().deviceSize(), stream_);
        ballTriangleInteractionMap_.alloc(getBallHandler().getBalls().deviceSize(), wallHandler_.getMeshWalls().deviceSize(), stream_);
    }

    void outputData() override
	{
		getBallHandler().outputBallVTU(getDir(), getFrame(), getStep(), getTime());
        wallHandler_.outputMeshWallVTU(getDir(), getFrame(), getStep(), getTime());
	}

	void neighborSearch() override
	{
		getBallHandler().neighborSearch(getGPUMaxThreadsPerBlock(), stream_);

        launchBallTriangleNeighborSearch(ballTriangleInteractions_, 
        ballTriangleInteractionMap_, 
        getBallHandler().getBalls(), 
        wallHandler_.getMeshWalls(), 
        triangleSpatialGrids_, 
        getGPUMaxThreadsPerBlock(), 
        stream_);
	}

	void integration1st(const double dt) override
	{
		getBallHandler().integration1st(getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
        wallHandler_.integration(dt, getGPUMaxThreadsPerBlock(), stream_);
	}

	void contactCalculation(const double dt) override
	{
        launchBallMeshWallInteractionCalculation(ballTriangleInteractions_, 
        getBallHandler().getBalls(), 
        wallHandler_.getMeshWalls(), 
        getContactModelParams(), 
        ballTriangleInteractionMap_, 
        dt, 
        getGPUMaxThreadsPerBlock(), 
        stream_);

		getBallHandler().contactCalculation(getContactModelParams(), dt, getGPUMaxThreadsPerBlock(), stream_);
	}

    cudaStream_t stream_;

    wallHandler wallHandler_;
    spatialGrid triangleSpatialGrids_;

    solidInteraction ballTriangleInteractions_;
    interactionMap ballTriangleInteractionMap_;
};