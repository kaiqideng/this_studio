#pragma once
#include "ballNeighborSearch.h"
#include "myStruct/wall.h"

extern "C" void updateTriGridCellStartEnd(spatialGrid& sptialGrids, 
int* hashIndex, 
int* hashValue, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition, 
const size_t numTri,
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchBallTriangleNeighborSearch(solidInteraction& ballTriangleInteractions, 
interactionMap &ballTriangleInteractionMap,
ball& balls, 
meshWall& meshWalls,
spatialGrid& triangleSpatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);