#include "externalForceTorque.h"

__global__ void addConstantForce(double3* force,
const double3* force_external,
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    force[idx] += force_external[idx];
}

extern "C" void launchAddConstantForceouble3(double3* force,
const double3* force_external,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addConstantForce <<<gridD, blockD, 0, stream>>> (force, 
    force_external,
    num);
}

__global__ void addGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double C_d,
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    double3 f = force[idx], t = torque[idx];
    force[idx] -= C_d * length(f) * normalize(velocity[idx]);
    torque[idx] -= C_d * length(t) * normalize(angularVelocity[idx]);
}

extern "C" void launchAddGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampCoeff,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addGlobalDampingForceTorque <<<gridD, blockD, 0, stream>>> (force, 
    torque, 
    velocity, 
    angularVelocity, 
    dampCoeff, 
    num);
}