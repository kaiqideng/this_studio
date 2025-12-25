#pragma once
#include "ballNeighborSearch.h"

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
SPH& SPHAndGhosts, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchSPHVirtualParticleNeighborSearch(SPHInteraction& SPHVirtualInteractions, 
interactionMap& SPHVirtualInteractionMap,
SPH& SPHAndGhosts, 
virtualParticle& virtualParticles, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);