#pragma once
#include "myContainer/myWall.h"
#include "myContainer/mySpatialGrid.h"
#include "myContainer/myUtility/myMat.h"
#include "integration.h"

class wallHandler
{
public:
    wallHandler(cudaStream_t s)
    {
        wallStream_ = s;
        infiniteWallsHostArrayChangedFlag_ = false;
    }

    ~wallHandler() = default;

    void addInfinitePlaneWall(double3 position, double3 normal, double3 velocity, int materialID)
    {
        if(!infiniteWallsHostArrayChangedFlag_)
        {
            infiniteWalls_.upload(wallStream_);
            infiniteWallsHostArrayChangedFlag_ = true;
        }
        infiniteWalls_.addPlane(position, velocity, normal, materialID);
    }

    void addInfiniteCylinderWall(double3 position, double3 velocity, double3 axis, double axisAngularVelocity, double radius, int materialID)
    {
        if(!infiniteWallsHostArrayChangedFlag_)
        {
            infiniteWalls_.upload(wallStream_);
            infiniteWallsHostArrayChangedFlag_ = true;
        }
        infiniteWalls_.addCylinder(position,velocity,axis,axisAngularVelocity,radius,materialID);
    }

protected:
    infiniteWall &infiniteWalls() {return infiniteWalls_;}

    void wallInitialize()
    {
        downloadWall();
    }

    void wallIntegrateBeforeContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        infiniteWallHalfIntegration(timeStep, maxThreadsPerBlock);
    }

    void wallIntegrateAfterContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        infiniteWallHalfIntegration(timeStep, maxThreadsPerBlock);
    }

private:
    void downloadWall()
    {
        if(infiniteWallsHostArrayChangedFlag_)
        {
            infiniteWalls_.download(wallStream_);
            infiniteWallsHostArrayChangedFlag_ = false;
        }
    } 

    void infiniteWallHalfIntegration(const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchInfiniteWallHalfIntegration(infiniteWalls_, timeStep, maxThreadsPerBlock, wallStream_);
    }

    cudaStream_t wallStream_;
    bool infiniteWallsHostArrayChangedFlag_;

    infiniteWall infiniteWalls_;
};