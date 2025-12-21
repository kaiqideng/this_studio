#pragma once
#include "ballNeighborSearch.h"

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
SPH& SPHs, 
SPHInteraction& SPHVirtualInteractions, 
interactionMap& SPHVirtualInteractionMap,
virtualParticle& virtualParticles, 
spatialGrid& SPHVirtualSpatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);