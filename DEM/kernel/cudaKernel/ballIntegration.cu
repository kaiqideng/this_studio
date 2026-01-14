#include"ballIntegration.h"

__global__ void calBallContactForceTorqueKernel(double3* contactForce, 
double3* contactTorque,
double3* contactPoint,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
int* objectPointed, 
int* objectPointing, 
double3* position, 
double3* velocity, 
double3* angularVelocity, 
const double* radius, 
const double* inverseMass, 
const int* materialID,
const double* hertzianE, 
const double* hertzianG, 
const double* hertzianRes, 
const double* hertzianK_r_k_s, 
const double* hertzianK_t_k_s, 
const double* hertzianMu_s, 
const double* hertzianMu_r, 
const double* hertzianMu_t, 
const double* linearK_n, 
const double* linearK_s, 
const double* linearK_r, 
const double* linearK_t, 
const double* linearD_n, 
const double* linearD_s, 
const double* linearD_r, 
const double* linearD_t, 
const double* linearMu_s, 
const double* linearMu_r, 
const double* linearMu_t, 
const size_t numMaterials,
const size_t contactParaArraySize,
const double dt,
const size_t numInteractions)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteractions) return;

	contactForce[idx] = make_double3(0, 0, 0);
	contactTorque[idx] = make_double3(0, 0, 0);

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];
	const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];
	const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double3 n_ij = normalize(r_i - r_j);
	const double delta = rad_i + rad_j - length(r_i - r_j);
	const double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
	const double rad_ij = rad_i * rad_j / (rad_i + rad_j);
	const double m_ij = 1. / (inverseMass[idx_i] + inverseMass[idx_j]);

	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;

	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	double3 epsilon_s = slidingSpring[idx];
	double3 epsilon_r = rollingSpring[idx];
	double3 epsilon_t = torsionSpring[idx];

    if (contactParaArraySize <= 0) return;
	const size_t param_ij = getContactParameterArraryIndex(materialID[idx_i], 
	materialID[idx_j], 
	numMaterials, 
	contactParaArraySize);

	if (linearK_n[param_ij] > 1.e-20)
	{
		LinearContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
		v_c_ij,
		w_ij,
		n_ij,
		delta,
		m_ij,
		rad_ij,
		dt,
		linearK_n[param_ij],
		linearK_s[param_ij],
		linearK_r[param_ij],
		linearK_t[param_ij],
		linearD_n[param_ij],
		linearD_s[param_ij],
		linearD_r[param_ij],
		linearD_t[param_ij],
		linearMu_s[param_ij],
		linearMu_r[param_ij],
		linearMu_t[param_ij]);
	}
	else
	{
		const double logR = log(hertzianRes[param_ij]);
		const double hertzianD = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
		v_c_ij,
		w_ij,
		n_ij,
		delta,
		m_ij,
		rad_ij,
		dt,
		hertzianD,
		hertzianE[param_ij],
		hertzianG[param_ij],
		hertzianK_r_k_s[param_ij],
		hertzianK_t_k_s[param_ij],
		hertzianMu_s[param_ij],
		hertzianMu_r[param_ij],
		hertzianMu_t[param_ij]);
	}

	contactForce[idx] = F_c;
	contactTorque[idx] = T_c;
	contactPoint[idx] = r_c;
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void calBondedForceTorqueKernel(double* normalForce, 
double* torsionTorque, 
double3* shearForce, 
double3* bendingTorque, 
double3* contactNormal, 
int* isBonded, 
int* objectPointed_b, 
int* objectPointing_b, 
double3* contactForce, 
double3* contactTorque, 
int* objectPointed, 
int* objectPointing, 
double3* force, 
double3* torque, 
double3* position, 
double3* velocity, 
double3* angularVelocity, 
const double* radius, 
const double* inverseMass, 
const int* materialID, 
int* interactionMapPrefixSumA,
const double* bondedGamma, 
const double* bondedE, 
const double* bondedK_n_k_s, 
const double* bondedSigma_s, 
const double* bondedC, 
const double* bondedMu,
const size_t numMaterials,
const size_t contactParaArraySize,
const double dt,
const size_t numBondedInteractions)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numBondedInteractions) return;

	if (isBonded[idx] == 0)
	{
		normalForce[idx] = 0;
		torsionTorque[idx] = 0;
		shearForce[idx] = make_double3(0, 0, 0);
		bendingTorque[idx] = make_double3(0, 0, 0);
		contactNormal[idx] = make_double3(0, 0, 0);
		return;
	}

	const int idx_i = objectPointed_b[idx];
	const int idx_j = objectPointing_b[idx];

	const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];
	const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double3 n_ij = normalize(r_i - r_j);
	const double3 n_ij0 = contactNormal[idx];
	const double delta = rad_i + rad_j - length(r_i - r_j);
	const double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));

	const size_t param_ij = getContactParameterArraryIndex(materialID[idx_i], materialID[idx_j], 
	numMaterials, contactParaArraySize);
    if (contactParaArraySize <= 0) return;

	double F_n = normalForce[idx];
	double3 F_s = shearForce[idx];
	double T_t = torsionTorque[idx];
	double3 T_b = bendingTorque[idx];
	isBonded[idx] = ParallelBondedContact(F_n, T_t, F_s, T_b,
	n_ij0,
	n_ij,
	v_c_ij,
	w_i,
	w_j,
	rad_i,
	rad_j,
	dt,
	bondedGamma[param_ij],
	bondedE[param_ij],
	bondedK_n_k_s[param_ij],
	bondedSigma_s[param_ij],
	bondedC[param_ij],
	bondedMu[param_ij]);

	normalForce[idx] = F_n;
	shearForce[idx] = F_s;
	torsionTorque[idx] = T_t;
	bendingTorque[idx] = T_b;
	contactNormal[idx] = n_ij;

	bool flag = false;
	int idx_c = 0;
	const int neighborStart_i = idx_i > 0 ? interactionMapPrefixSumA[idx_i - 1] : 0;
	const int neighborEnd_i = interactionMapPrefixSumA[idx_i];
	for (int k = neighborStart_i; k < neighborEnd_i; k++)
	{
		if (objectPointing[k] == idx_j)
		{
			flag = true;
			idx_c = k;
			break;
		}
	}
	if (!flag)
	{
		atomicAddDouble3(force, idx_i, F_n * n_ij + F_s);
		atomicAddDouble3(torque, idx_i, T_t * n_ij + T_b + cross(r_c - r_i, F_s));
		atomicAddDouble3(force, idx_j, -F_n * n_ij - F_s);
		atomicAddDouble3(torque, idx_j, -T_t * n_ij - T_b + cross(r_c - r_j, -F_s));
		return;
	}

	contactForce[idx_c] += F_n * n_ij + F_s;
	contactTorque[idx_c] += T_t * n_ij + T_b;
}

__global__ void sumForceTorqueFromInteractionKernel(double3* contactForce, 
double3* contactTorque, 
double3* contactPoint,
int* objectPointed, 
int* objectPointing, 
double3* force, 
double3* torque, 
double3* position, 
int* interactionMapHashIndex,
int* interactionMaPrefixSumA, 
int* interactionMapStartB, 
int* interactionMapEndB,
const size_t numBalls)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBalls) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0, 0, 0);
	double3 T_i = make_double3(0, 0, 0);
	for (int k = idx_i > 0 ? interactionMaPrefixSumA[idx_i - 1] : 0; k < interactionMaPrefixSumA[idx_i]; k++)
	{
		double3 r_c = contactPoint[k];
		F_i += contactForce[k];
		T_i += contactTorque[k] + cross(r_c - r_i, contactForce[k]);
	}

	if (interactionMapStartB[idx_i] != 0xFF)
	{
		for (int k = interactionMapStartB[idx_i]; k < interactionMapEndB[idx_i]; k++)
		{
			int k1 = interactionMapHashIndex[k];
			double3 r_c = contactPoint[k1];
			F_i -= contactForce[k1];
			T_i -= contactTorque[k1];
			T_i -= cross(r_c - r_i, contactForce[k1]);
		}
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}

__global__ void sumClumpForceTorqueKernel(double3* force_c, 
double3* torque_c, 
double3* position_c, 
const int* pebbleStartIndex, 
const int* pebbleEndIndex,
double3* force_p, 
double3* torque_p, 
double3* position_p,
const size_t numClumps)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClumps) return;
	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	for (int i = pebbleStartIndex[idx_c]; i < pebbleEndIndex[idx_c]; i++)
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
double3* force, 
double3* torque, 
const double* radius, 
const double* invMass, 
const int* clumpID, 
const double3 g,
const double dt,
const size_t numBalls)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBalls) return;

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
double3* position_c, 
double3* force_c, 
double3* torque_c, 
const double* invMass_c, 
quaternion* orientation, 
const symMatrix* inverseInertiaTensor, 
const int* pebbleStartIndex, 
const int* pebbleEndIndex,
double3* velocity_p, 
double3* angularVelocity_p, 
double3* position_p, 
const double3 g,
const double dt, 
const size_t numClumps)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClumps) return;

    double invM_c = invMass_c[idx_c];
	double3 w_c = make_double3(0.0, 0.0, 0.0);
	if (invM_c > 0.) 
	{
		velocity_c[idx_c] += (force_c[idx_c] * invM_c + g) * dt;
		angularVelocity_c[idx_c] += (rotateInverseInertiaTensor(orientation[idx_c], inverseInertiaTensor[idx_c]) * torque_c[idx_c]) * dt;
		w_c = angularVelocity_c[idx_c];
	}
	for (size_t i = pebbleStartIndex[idx_c]; i < pebbleEndIndex[idx_c]; i++)
	{
		double3 r_pc = position_p[i] - position_c[idx_c];
		velocity_p[i] = velocity_c[idx_c] + cross(w_c, r_pc);
		angularVelocity_p[i] = w_c;
	}
}

__global__ void orientationIntegrationKernel(quaternion* orientation, 
double3* angularVelocity, 
const double dt, 
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	orientation[idx] = quaternionRotate(orientation[idx], angularVelocity[idx], dt);
}

__global__ void positionIntegrationKernel(double3* position, double3* velocity, 
const double dt,
const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	position[idx_i] += dt * velocity[idx_i];
}

extern "C" void launchBall1stHalfIntegration(ball& balls, 
const double3 gravity, 
const double timeStep, 
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
	ballVelocityAngularVelocityIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (balls.velocity(), 
	balls.angularVelocity(), 
	balls.force(), 
	balls.torque(), 
	balls.radius(), 
	balls.inverseMass(), 
	balls.clumpID(), 
	gravity, 
	0.5 * timeStep, 
	balls.deviceSize());

	positionIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (balls.position(), 
	balls.velocity(), 
	timeStep,
	balls.deviceSize());

	CUDA_CHECK(cudaMemsetAsync(balls.force(), 0, balls.deviceSize() * sizeof(double3), stream));
	CUDA_CHECK(cudaMemsetAsync(balls.torque(), 0, balls.deviceSize() * sizeof(double3), stream));
}

extern "C" void launchBall2ndHalfIntegration(ball& balls, 
const double3 gravity, 
const double timeStep, 
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
	ballVelocityAngularVelocityIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (balls.velocity(), 
    balls.angularVelocity(), 
	balls.force(), 
    balls.torque(), 
    balls.radius(), 
    balls.inverseMass(), 
    balls.clumpID(), 
	gravity, 
    0.5 * timeStep, 
    balls.deviceSize());
}

extern "C" void launchClump1stHalfIntegration(clump& clumps, 
ball& balls, 
const double3 gravity, 
const double timeStep, 
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
	clumpVelocityAngularVelocityIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (clumps.velocity(),
	clumps.angularVelocity(),
	clumps.position(),
	clumps.force(),
	clumps.torque(),
	clumps.inverseMass(),
	clumps.orientation(),
	clumps.inverseInertiaTensor(),
	clumps.pebbleStart(),
	clumps.pebbleEnd(),
	balls.velocity(),
	balls.angularVelocity(),
	balls.position(),
	gravity,
	0.5 * timeStep,
	clumps.deviceSize());

	orientationIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (clumps.orientation(), 
	clumps.angularVelocity(), 
	timeStep, 
	clumps.deviceSize());

	positionIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (clumps.position(), 
    clumps.velocity(), 
    timeStep,
    clumps.deviceSize());

	CUDA_CHECK(cudaMemsetAsync(clumps.force(), 0, clumps.deviceSize() * sizeof(double3), stream));
    CUDA_CHECK(cudaMemsetAsync(clumps.torque(), 0, clumps.deviceSize() * sizeof(double3), stream));
}

extern "C" void launchClump2ndHalfIntegration(clump& clumps, 
ball& balls, 
const double3 gravity, 
const double timeStep, 
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
	sumClumpForceTorqueKernel <<<gridDim, blockDim, 0, stream>>> (clumps.force(), 
	clumps.torque(), 
	clumps.position(), 
	clumps.pebbleStart(), 
	clumps.pebbleEnd(), 
	balls.force(), 
	balls.torque(), 
	balls.position(), 
	clumps.deviceSize());

	clumpVelocityAngularVelocityIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (clumps.velocity(),
	clumps.angularVelocity(),
	clumps.position(),
	clumps.force(),
	clumps.torque(),
	clumps.inverseMass(),
	clumps.orientation(),
	clumps.inverseInertiaTensor(),
	clumps.pebbleStart(),
	clumps.pebbleEnd(),
	balls.velocity(),
	balls.angularVelocity(),
	balls.position(),
	gravity,
	0.5 * timeStep,
	clumps.deviceSize());
}

extern "C" void launchBallContactCalculation(solidInteraction &ballInteractions, 
bondedInteraction &bondedBallInteractions, 
ball &balls, 
contactModelParameters &contactModelParams,
interactionMap &ballInteractionMap,
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    size_t gridDim = 1, blockDim = 1;
	if (setGPUGridBlockDim(gridDim, blockDim, ballInteractions.activeSize(), maxThreadsPerBlock))
	{
		calBallContactForceTorqueKernel <<<gridDim, blockDim, 0, stream>>> (ballInteractions.force(),
		ballInteractions.torque(),
		ballInteractions.contactPoint(),
		ballInteractions.slidingSpring(),
		ballInteractions.rollingSpring(),
		ballInteractions.torsionSpring(),
		ballInteractions.objectPointed(),
		ballInteractions.objectPointing(),
		balls.position(),
		balls.velocity(),
		balls.angularVelocity(),
		balls.radius(),
		balls.inverseMass(),
		balls.materialID(),
		contactModelParams.hertzian.effectiveYoungsModulus,
		contactModelParams.hertzian.effectiveShearModulus,
		contactModelParams.hertzian.restitutionCoefficient,
		contactModelParams.hertzian.rollingStiffnessToShearStiffnessRatio,
		contactModelParams.hertzian.torsionStiffnessToShearStiffnessRatio,
		contactModelParams.hertzian.slidingFrictionCoefficient,
		contactModelParams.hertzian.rollingFrictionCoefficient,
		contactModelParams.hertzian.torsionFrictionCoefficient,
		contactModelParams.linear.normalStiffness,
		contactModelParams.linear.slidingStiffness,
		contactModelParams.linear.rollingStiffness,
		contactModelParams.linear.torsionStiffness,
		contactModelParams.linear.normalDampingCoefficient,
		contactModelParams.linear.slidingDampingCoefficient,
		contactModelParams.linear.rollingDampingCoefficient,
		contactModelParams.linear.torsionDampingCoefficient,
		contactModelParams.linear.slidingFrictionCoefficient,
		contactModelParams.linear.rollingFrictionCoefficient,
		contactModelParams.linear.torsionFrictionCoefficient,
		contactModelParams.numberOfMaterials,
		contactModelParams.pairTableSize,
		timeStep,
		ballInteractions.activeSize());
	}

	if (setGPUGridBlockDim(gridDim, blockDim, bondedBallInteractions.deviceSize(), maxThreadsPerBlock))
	{
		calBondedForceTorqueKernel <<<gridDim, blockDim, 0, stream>>> (bondedBallInteractions.normalForce(),
		bondedBallInteractions.torsionTorque(),
		bondedBallInteractions.shearForce(),
		bondedBallInteractions.bendingTorque(),
		bondedBallInteractions.contactNormal(),
		bondedBallInteractions.isBonded(),
		bondedBallInteractions.objectPointed(),
		bondedBallInteractions.objectPointing(),
		ballInteractions.force(),
		ballInteractions.torque(),
		ballInteractions.objectPointed(),
		ballInteractions.objectPointing(),
		balls.force(),
		balls.torque(),
		balls.position(),
		balls.velocity(),
		balls.angularVelocity(),
		balls.radius(),
		balls.inverseMass(),
		balls.materialID(),
		ballInteractionMap.prefixSumA(),
		contactModelParams.bonded.bondRadiusMultiplier,
		contactModelParams.bonded.bondYoungsModulus,
		contactModelParams.bonded.normalToShearStiffnessRatio,
		contactModelParams.bonded.tensileStrength,
		contactModelParams.bonded.cohesion,
		contactModelParams.bonded.frictionCoefficient,
		contactModelParams.numberOfMaterials,
		contactModelParams.pairTableSize,
		timeStep,
		bondedBallInteractions.deviceSize());
	}
	
	if (setGPUGridBlockDim(gridDim, blockDim, balls.deviceSize(), maxThreadsPerBlock))
	{
		sumForceTorqueFromInteractionKernel <<<gridDim, blockDim, 0, stream>>> (ballInteractions.force(),
		ballInteractions.torque(),
		ballInteractions.contactPoint(),
		ballInteractions.objectPointed(),
		ballInteractions.objectPointing(),
		balls.force(),
		balls.torque(),
		balls.position(),
		ballInteractionMap.hashIndex(),
		ballInteractionMap.prefixSumA(),
		ballInteractionMap.startB(),
		ballInteractionMap.endB(),
		balls.deviceSize());
	}
}