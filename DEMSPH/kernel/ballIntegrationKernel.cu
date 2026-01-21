#include "ballIntegrationKernel.h"

__global__ void sumClumpForceTorqueKernel(double3* force_c, 
double3* torque_c, 
const double3* position_c, 
const int* pebbleStart_c, 
const int* pebbleEnd_c,
const double3* force_p, 
const double3* torque_p, 
const double3* position_p,
const size_t numClump)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClump) return;
	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	for (int i = pebbleStart_c[idx_c]; i < pebbleEnd_c[idx_c]; i++)
	{
		double3 r_i = position_p[i];
		double3 F_i = force_c[i];
		double3 r_c = position_c[idx_c];
		F_c += F_i;
		T_c += torque_p[i] + cross(r_i - r_c, F_i);
	}

	force_c[idx_c] += F_c;
	torque_c[idx_c] += T_c;
}

__global__ void ballVelocityAngularVelocityIntegrationKernel(double3* velocity, 
double3* angularVelocity, 
const double3* force, 
const double3* torque, 
const double* radius, 
const double* invMass, 
const int* clumpID, 
const double3 g,
const double dt,
const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

	if (clumpID[idx_i] >= 0) return;

	double invM_i = invMass[idx_i];
	if (invM_i < 1.e-20) return;

	velocity[idx_i] += (force[idx_i] * invM_i + g) * dt;

	double rad_i = radius[idx_i];
	if (rad_i < 1.e-20) return;
	double I_i = 0.4 * rad_i * rad_i / invM_i;

	angularVelocity[idx_i] += torque[idx_i] / I_i * dt;
}

__global__ void clumpVelocityAngularVelocityIntegrationKernel(double3* velocity_c, 
double3* angularVelocity_c, 
double3* velocity_p, 
double3* angularVelocity_p, 
const double3* position_c, 
const double3* force_c, 
const double3* torque_c, 
const double* invMass_c, 
const quaternion* orientation_c, 
const symMatrix* inverseInertiaTensor_c, 
const int* pebbleStart_c, 
const int* pebbleEnd_c,
const double3* position_p, 
const double3 g,
const double dt, 
const size_t numClump)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClump) return;

    double invM_c = invMass_c[idx_c];
	double3 w_c = make_double3(0.0, 0.0, 0.0);
	if (invM_c > 0.) 
	{
		velocity_c[idx_c] += (force_c[idx_c] * invM_c + g) * dt;
		angularVelocity_c[idx_c] += (rotateInverseInertiaTensor(orientation_c[idx_c], inverseInertiaTensor_c[idx_c]) * torque_c[idx_c]) * dt;
		w_c = angularVelocity_c[idx_c];
	}
	for (size_t i = pebbleStart_c[idx_c]; i < pebbleEnd_c[idx_c]; i++)
	{
		double3 r_pc = position_p[i] - position_c[idx_c];
		velocity_p[i] = velocity_c[idx_c] + cross(w_c, r_pc);
		angularVelocity_p[i] = w_c;
	}
}

__global__ void orientationIntegrationKernel(quaternion* orientation, 
const double3* angularVelocity, 
const double dt, 
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	orientation[idx] = quaternionRotate(orientation[idx], angularVelocity[idx], dt);
}

__global__ void positionIntegrationKernel(double3* position, 
const double3* velocity, 
const double dt,
const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	position[idx_i] += dt * velocity[idx_i];
}

extern "C" void launchBall1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* radius, 
double* invMass, 
int* clumpID, 

const double3 gravity, 
const double timeStep,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	ballVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	radius, 
	invMass, 
	clumpID, 
	gravity,
	0.5 * timeStep,
	numBall);

	positionIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	velocity, 
	timeStep,
	numBall);

	cudaMemsetAsync(force, 0, numBall * sizeof(double3), stream);
    cudaMemsetAsync(torque, 0, numBall * sizeof(double3), stream);
}

extern "C" void launchBall2ndHalfIntegration(double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* radius, 
double* invMass, 
int* clumpID,

const double3 gravity, 
const double timeStep, 

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	ballVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	radius, 
	invMass, 
	clumpID, 
	gravity,
	0.5 * timeStep,
	numBall);
}

extern "C" void launchClump1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 
const int* pebbleStart, 
const int* pebbleEnd,

double3* position_p, 
double3* velocity_p, 
double3* angularVelocity_p, 

const double3 gravity, 
const double timeStep,

const size_t numClump,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	clumpVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	velocity_p, 
    angularVelocity_p,
	position, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor, 
    pebbleStart, 
    pebbleEnd,
	position_p,
	gravity,
	0.5 * timeStep,
	numClump);

	positionIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	velocity, 
	timeStep,
	numClump);

	orientationIntegrationKernel <<<gridD, blockD, 0, stream>>> (orientation,
	angularVelocity,
	timeStep,
	numClump);

	cudaMemsetAsync(force, 0, numClump * sizeof(double3), stream);
    cudaMemsetAsync(torque, 0, numClump * sizeof(double3), stream);
}

extern "C" void launchClump2ndHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 
const int* pebbleStart, 
const int* pebbleEnd,

double3* position_p, 
double3* velocity_p, 
double3* angularVelocity_p, 
double3* force_p, 
double3* torque_p,

const double3 gravity, 
const double timeStep,

const size_t numClump,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	sumClumpForceTorqueKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque,
	position,
	pebbleStart,
	pebbleEnd,
	force_p,
	torque_p,
	position_p,
	numClump);

	clumpVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	velocity_p, 
    angularVelocity_p,
	position, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor, 
    pebbleStart, 
    pebbleEnd,
	position_p,
	gravity,
	0.5 * timeStep,
	numClump);
}