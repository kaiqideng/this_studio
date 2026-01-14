#pragma once
#include "SPHNeighborSearch.h"
#include "ballIntegration.h"

// Wendland 5th-order (C2) kernel in 3D
__device__ __forceinline__ double wendlandKernel3D(double r, double h)
{
	double q = r / h;
	if (q >= 2.0) return 0.0;

	double sigma = 21.0 / (16.0 * pi() * h * h * h);
	double term = 1.0 - 0.5 * q;
	return sigma * pow(term, 4) * (1.0 + 2.0 * q);
}

__device__ __forceinline__ double3 gradWendlandKernel3D(const double3& rij, double h)
{
	double r = length(rij);
	if (r < 1.e-20 || r >= 2.0 * h) return make_double3(0, 0, 0);

	double q = r / h;
	double sigma = 21.0 / (16.0 * pi() * h * h * h);

	double term = 1.0 - 0.5 * q;
	// d/dq of [ (1 - q/2)^4 (1 + 2q) ]
	double dW_dq = (-2.0 * pow(term, 3) * (1.0 + 2.0 * q) + 2.0 * pow(term, 4));

	// chain rule: dW/dr = (dW/dq) * (dq/dr) = (dW/dq) / h
	double dWdr = sigma * dW_dq / h;

	double factor = dWdr / r;  // multiply with rij/r
	return factor * rij;
}

extern "C" void launchCalDummyParticleNormal(WCSPH& WCSPHs, 
SPHInteraction& SPHInteractions,
interactionMap& SPHInteractionMap,
const size_t maxThreadPerBlock,
cudaStream_t stream);

extern "C" void launchWCSPH1stIntegration(WCSPH& WCSPHs, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream);

extern "C" void launchWCSPH2ndIntegration(WCSPH& WCSPHs, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream);

extern "C" void launchISPH1stIntegration(ISPH& ISPHs, 
SPHInteraction& SPHInteractions, 
interactionMap &SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim,
cudaStream_t stream);

extern "C" void launchISPH2ndIntegration(ISPH& ISPHs, 
SPHInteraction& SPHInteractions, 
interactionMap &SPHInteractionMap,
const double timeStep,
const size_t gridDim,
const size_t blockDim,
cudaStream_t stream);