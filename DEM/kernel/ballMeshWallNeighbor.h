#pragma once
#include "ballNeighborSearch.h"
#include"wallHandler.h"

extern "C" void launchBallTriangleNeighborSearch(solidInteraction& ballTriangleInteractions, 
interactionMap &ballTriangleInteractionMap,
ball& balls, 
meshWall& meshWalls,
spatialGrid& triangleSpatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);