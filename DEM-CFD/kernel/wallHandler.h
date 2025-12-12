#pragma once
#include "myContainer/myUtility/myQua.h"
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
        triangleWallsHostArrayChangedFlag_ = false;
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
        infiniteWalls_.addCylinder(position,velocity, axis,axisAngularVelocity, radius,materialID);
    }

    void addTriangleWall(const std::vector<double3> &vertex, const std::vector<int3> &triIndices, const double3 &posistion, const double3 &velocity, const double3 &angularVelocity, int matirialID)
    {
        if(!triangleWallsHostArrayChangedFlag_)
        {
            triangleWalls_.upload(wallStream_);
            triangleWallsHostArrayChangedFlag_ = true;
        }
        triangleWalls_.addWallFromMesh(vertex, triIndices, posistion, velocity, angularVelocity, make_quaternion(1, 0, 0, 0), matirialID);
    }

protected:
    infiniteWall &infiniteWalls() {return infiniteWalls_;}

    triangleWall &triangleWalls() {return triangleWalls_;}

    spatialGrid &triangleSpatialGrids() {return triangleSpatialGrids_;}

    void wallInitialize(const double3 domainOrigin, const double3 domainSize)
    {
        downloadWall(domainOrigin, domainSize);
    }

    void wallIntegrateBeforeContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        infiniteWallHalfIntegration(timeStep, maxThreadsPerBlock);
        triangleWallHalfIntegration(timeStep, maxThreadsPerBlock);
    }

    void wallIntegrateAfterContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        infiniteWallHalfIntegration(timeStep, maxThreadsPerBlock);
        triangleWallHalfIntegration(timeStep, maxThreadsPerBlock);
    }

private:
    void downloadWall(const double3 domainOrigin, const double3 domainSize)
    {
        if(infiniteWallsHostArrayChangedFlag_)
        {
            infiniteWalls_.download(wallStream_);
            infiniteWallsHostArrayChangedFlag_ = false;
        }
        if(triangleWallsHostArrayChangedFlag_)
        {
            triangleWalls_.download(wallStream_);
            double cellSizeOneDim = 1.25 * triangleWalls_.getMaxEdgeLength();
            triangleSpatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, wallStream_);
            triangleWallsHostArrayChangedFlag_ = false;
        }
    } 

    void infiniteWallHalfIntegration(const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchInfiniteWallHalfIntegration(infiniteWalls_, timeStep, maxThreadsPerBlock, wallStream_);
    }

    void triangleWallHalfIntegration(const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchTriangleWallHalfIntegration(triangleWalls_,timeStep,maxThreadsPerBlock,wallStream_);
    }

    cudaStream_t wallStream_;
    bool infiniteWallsHostArrayChangedFlag_;
    bool triangleWallsHostArrayChangedFlag_;

    infiniteWall infiniteWalls_;
    triangleWall triangleWalls_;
    spatialGrid triangleSpatialGrids_;
};