#pragma once
#include "ballNeighborSearch.h"

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap &SPHInteractionMap,
SPH& SPHs, 
spatialGrid& SPHSpatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);