#include "wallIntegrationKernel.h"

__global__ void wallOrientationIntegrationKernel(quaternion* orientation, 
const double3* angularVelocity, 
const double dt, 
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	orientation[idx] = quaternionRotate(orientation[idx], angularVelocity[idx], dt);
}

__global__ void wallPositionIntegrationKernel(double3* position, 
const double3* velocity, 
const double dt,
const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	position[idx_i] += dt * velocity[idx_i];
}

__global__ void updateWallVertexGlobalPosition(double3* globalPosition_v,
const double3* localPosition_v,
const int* wallIndex_v,
const double3* position,
const quaternion* orientation,
const size_t numVertex)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numVertex) return;

    int idx_w = wallIndex_v[idx];
    globalPosition_v[idx] = position[idx_w] + rotateVectorByQuaternion(orientation[idx_w], localPosition_v[idx]);
}

extern "C" void launchWallIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity,
quaternion* orientation, 
const double timeStep,
const size_t numWall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
    wallPositionIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	velocity, 
	timeStep,
	numWall);

	wallOrientationIntegrationKernel <<<gridD, blockD, 0, stream>>> (orientation,
	angularVelocity,
	timeStep,
	numWall);
}

extern "C" void launchUpdateWallVertexGlobalPosition(double3* globalPosition_v,
double3* localPosition_v,
int* wallIndex_v,

double3* position_w,
quaternion* orientation_w,

const size_t numVertex,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
    updateWallVertexGlobalPosition <<<gridD, blockD, 0, stream>>> (globalPosition_v,
    localPosition_v,
    wallIndex_v,
    position_w,
    orientation_w,
    numVertex);
}