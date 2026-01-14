#pragma once
#include "ballNeighborSearch.h"
#include "myStruct/SPH.h"

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
spatialGrid& spatialGrids, 
int* SPHHashIndex,
int* SPHHashValue,
double3* SPHPosition,
const double* SPHSmoothLength,
const size_t num,
const size_t maxThreadsPerBlock,
cudaStream_t stream);