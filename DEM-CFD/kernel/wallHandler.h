#pragma once
#include "myContainer/myWall.h"
#include "myContainer/mySpatialGrid.h"
#include "myContainer/myUtility/myMat.h"
#include <vector>

class wallHandler
{
public:
    wallHandler(cudaStream_t s)
    {
        wallStream = s;
        infiniteWallsHostArrayChangedFlag = false;
    }

    ~wallHandler() = default;

    void addInfinitePlaneWall(double3 position, double3 normal, double3 velocity, int materialID)
    {
        if(!infiniteWallsHostArrayChangedFlag)
        {
            infiniteWalls.upload(wallStream);
            infiniteWallsHostArrayChangedFlag = true;
        }
        infiniteWalls.addPlane(position, velocity, normal, materialID);
    }

    void addInfiniteCylinderWall(double3 position, double3 velocity, double3 axis, double axisAngularVelocity, double radius, int materialID)
    {
        if(!infiniteWallsHostArrayChangedFlag)
        {
            infiniteWalls.upload(wallStream);
            infiniteWallsHostArrayChangedFlag = true;
        }
        infiniteWalls.addCylinder(position,velocity,axis,axisAngularVelocity,radius,materialID);
    }

private:
    void downloadWall()
    {
        if(infiniteWallsHostArrayChangedFlag)
        {
            infiniteWalls.download(wallStream);
            infiniteWallsHostArrayChangedFlag = false;
        }
    } 

    cudaStream_t wallStream;
    bool infiniteWallsHostArrayChangedFlag;

    infiniteWall infiniteWalls;
};