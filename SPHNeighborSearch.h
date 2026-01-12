#pragma once
#include "DEM/kernel/cudaKernel/ballNeighborSearch.h"
#include "SPH.h"

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
spatialGrid& spatialGrids, 
int* SPHHashIndex,
int* SPHHashValue,
double3* SPHPosition,
const double* SPHSmoothLength,
const size_t numSPHs,
const size_t numGhosts,
const size_t maxThreadsPerBlock,
cudaStream_t stream);