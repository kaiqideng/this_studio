#pragma once
#include "buildHashStartEnd.h"
#include "myStruct/interaction.h"
#include "myStruct/particle.h"
#include "myStruct/spatialGrid.h"

extern "C" void launchBallNeighborSearch(solidInteraction& ballInteractions, 
interactionMap &ballInteractionMap,
ball& balls, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);