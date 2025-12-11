#include "integration.h"
#include "myContainer/myHash.h"
#include "myContainer/myUtility/myQua.h"
#include "myContainer/myUtility/myVec.h"
#include "myContainer/myWall.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)       // sm 6.0+
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	return atomicAdd(addr, val);
}
#else                                                   
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	auto  addr_ull = reinterpret_cast<unsigned long long*>(addr);
	unsigned long long old = *addr_ull, assumed;

	do {
		assumed = old;
		double  old_d = __longlong_as_double(assumed);
		double  new_d = old_d + val;
		old = atomicCAS(addr_ull, assumed, __double_as_longlong(new_d));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

__device__ __forceinline__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v)
{
    atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}

__global__ void calSolidParticleContactForceTorqueKernel(double3* contactForce, double3* contactTorque,
	double3* slidingSpring, double3* rollingSpring, double3* torsionSpring, int* objectPointed, int* objectPointing, 
    double3* position, double3* velocity, double3* angularVelocity, double* radius, double* inverseMass, int* materialID,
	double* hertzianE, double* hertzianG, double* hertzianRes, double* hertzianK_r_k_s, double* hertzianK_t_k_s, 
	double* hertzianMu_s, double* hertzianMu_r, double* hertzianMu_t, 
	double* linearK_n, double* linearK_s, double* linearK_r, double* linearK_t, 
	double* linearD_n, double* linearD_s, double* linearD_r, double* linearD_t, double* linearMu_s, double* linearMu_r, double* linearMu_t, 
	const size_t numMaterials,
	const size_t contactParaArraySize,
	const double dt,
    const size_t numInteractions)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteractions) return;

	//!!!
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

	const size_t param_ij = getContactParameterArraryIndex(materialID[idx_i], materialID[idx_j], 
	numMaterials, contactParaArraySize);
	if (contactParaArraySize <= 0) return;

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
		const double D = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij,
			w_ij,
			n_ij,
			delta,
			m_ij,
			rad_ij,
			dt,
			D,
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
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void calSolidParticleInfiniteWallContactForceTorqueKernel(double3* contactForce, double3* contactTorque,
	double3* slidingSpring, double3* rollingSpring, double3* torsionSpring, int* objectPointed, int* objectPointing, 
    double3* position, double3* velocity, double3* angularVelocity, double* radius, double* inverseMass, int* materialID,
	double3* position_iw, double3* velocity_iw, double3* axis_iw, double* axisAngularVelocity_iw, double* radius_iw, int* materialID_iw,
	double* hertzianE, double* hertzianG, double* hertzianRes, double* hertzianK_r_k_s, double* hertzianK_t_k_s, 
	double* hertzianMu_s, double* hertzianMu_r, double* hertzianMu_t, 
	double* linearK_n, double* linearK_s, double* linearK_r, double* linearK_t, 
	double* linearD_n, double* linearD_s, double* linearD_r, double* linearD_t, double* linearMu_s, double* linearMu_r, double* linearMu_t, 
	const size_t numMaterials,
	const size_t contactParaArraySize,
	const double dt,
    const size_t numInteractions)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteractions) return;

    //!!!
	contactForce[idx] = make_double3(0, 0, 0);
	contactTorque[idx] = make_double3(0, 0, 0);

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];
	const double rad_i = radius[idx_i];
	const double rad_j = radius_iw[idx_j];
	const double3 r_i = position[idx_i];
	const double3 r_j = position_iw[idx_j];

    const double3 n = normalize(axis_iw[idx_j]);
	const double t = dot(r_i - r_j, n);
    double delta = rad_i - fabs(t);
	double3 n_ij = n;
	if(t < 0) n_ij = -n;
	double3 r_c = r_i - n_ij * (rad_i - delta);
	double rad_ij = rad_i;
    if(rad_j > 1.e-20)
    {
        double3 p = dot(r_i - r_j,n) * n + r_j;
        delta = rad_i + rad_j - length(r_i - p) ;
		n_ij = normalize(r_i - p);
        if(delta > rad_i) delta = length(r_i - p) + rad_i - rad_j;// inside
		r_c = r_i - n_ij * (rad_i - 0.5 * delta);
		rad_ij = rad_i * rad_j / (rad_i + rad_j);
    }

	const double m_ij = 1. / inverseMass[idx_i];

	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity_iw[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = axisAngularVelocity_iw[idx_j] * n;
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;

	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	double3 epsilon_s = slidingSpring[idx];
	double3 epsilon_r = rollingSpring[idx];
	double3 epsilon_t = torsionSpring[idx];

	const size_t param_ij = getContactParameterArraryIndex(materialID[idx_i], materialID_iw[idx_j], 
	numMaterials, contactParaArraySize);
	if (contactParaArraySize <= 0) return;

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
		const double D = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij,
			w_ij,
			n_ij,
			delta,
			m_ij,
			rad_ij,
			dt,
			D,
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
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void calBondedForceTorqueKernel(double* normalForce, double* torsionTorque, double3* shearForce, double3* bendingTorque, 
    double3* contactNormal, int* isBonded, int* objectPointed_b, int* objectPointing_b, 
    double3* contactForce, double3* contactTorque, int* objectPointed, int* objectPointing, 
    double3* force, double3* torque, double3* position, double3* velocity, double3* angularVelocity, 
	double* radius, double* inverseMass, int* materialID, int* solidParticleNeighborPrefixSum,
	double* bondedGamma, double* bondedE, double* bondedK_n_k_s, double* bondedSigma_s, double* bondedC, double* bondedMu,
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

	size_t param_ij = getContactParameterArraryIndex(materialID[idx_i], materialID[idx_j], 
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

	bool foundInteractions = false;
	int idx_c = 0;
	const int neighborStart_i = idx_i > 0 ? solidParticleNeighborPrefixSum[idx_i - 1] : 0;
	const int neighborEnd_i = solidParticleNeighborPrefixSum[idx_i];
	for (int k = neighborStart_i; k < neighborEnd_i; k++)
	{
		if (objectPointing[k] == idx_j)
		{
			foundInteractions = true;
			idx_c = k;
			break;
		}
	}
	if (!foundInteractions)
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

__global__ void sumForceTorqueFromSolidParticleInteractionKernel(double3* contactForce, double3* contactTorque, 
    int* objectPointed, int* objectPointing, int* interactionHashIndex,
    double3* force, double3* torque, double3* position, double* radius, 
	int* solidParticleNeighborPrefixSum, int* solidParticleInteractionIndexRangeStart, int* solidParticleInteractionIndexRangeEnd,
	const size_t numParticles)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numParticles) return;

	double rad_i = radius[idx_i];
	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0, 0, 0);
	double3 T_i = make_double3(0, 0, 0);
	for (int k = idx_i > 0 ? solidParticleNeighborPrefixSum[idx_i - 1] : 0; k < solidParticleNeighborPrefixSum[idx_i]; k++)
	{
		int idx_j = objectPointing[k];
		double rad_j = radius[idx_j];
		double3 r_j = position[idx_j];
		double3 n_ij = normalize(r_i - r_j);
		double delta = rad_i + rad_j - length(r_i - r_j);
		double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
		F_i += contactForce[k];
		T_i += contactTorque[k] + cross(r_c - r_i, contactForce[k]);
	}

	if (solidParticleInteractionIndexRangeStart[idx_i] != 0xFF)
	{
		for (int k = solidParticleInteractionIndexRangeStart[idx_i]; k < solidParticleInteractionIndexRangeEnd[idx_i]; k++)
		{
			int k1 = interactionHashIndex[k];
			int idx_j = objectPointed[k1];
			double rad_j = radius[idx_j];
			double3 r_j = position[idx_j];
			double3 n_ij = normalize(r_i - r_j);
			double delta = rad_i + rad_j - length(r_i - r_j);
			double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
			F_i -= contactForce[k1];
			T_i -= contactTorque[k1];
			T_i -= cross(r_c - r_i, contactForce[k1]);
		}
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}

__global__ void sumForceTorqueFromSolidParticleInfiniteWallInteractionKernel(double3* contactForce, double3* contactTorque, 
    int* objectPointing, 
    double3* force, double3* torque, double3* position, double* radius, 
	int* solidParticleNeighborPrefixSum, 
	double3* position_iw, double3* axis_iw, double* radius_iw, 
	const size_t numParticles)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numParticles) return;

	double rad_i = radius[idx_i];
	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0, 0, 0);
	double3 T_i = make_double3(0, 0, 0);
	for (int k = idx_i > 0 ? solidParticleNeighborPrefixSum[idx_i - 1] : 0; k < solidParticleNeighborPrefixSum[idx_i]; k++)
	{
		int idx_j = objectPointing[k];
		double rad_j = radius_iw[idx_j];
		double3 r_j = position_iw[idx_j];

        double3 n = normalize(axis_iw[idx_j]);
		double t = dot(r_i - r_j, n);
        double delta = rad_i - fabs(t);
		double3 n_ij = n;
		if(t < 0) n_ij = -n;
		double3 r_c = r_i - n_ij * (rad_i - delta);
        if(rad_j > 1.e-20)
        {
            double3 p = dot(r_i - r_j,n) * n + r_j;
            delta = rad_i + rad_j - length(r_i - p) ;
			n_ij = normalize(r_i - p);
            if(delta > rad_i) delta = length(r_i - p) + rad_i - rad_j;
			r_c = r_i - n_ij * (rad_i - 0.5 * delta);
        }

		F_i += contactForce[k];
		T_i += contactTorque[k] + cross(r_c - r_i, contactForce[k]);
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}

__global__ void sumClumpForceTorqueKernel(double3* force_c, double3* torque_c, double3* position_c, 
    int* pebbleStartIndex, int* pebbleEndIndex,
	double3* force_p, double3* torque_p, double3* position_p,
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
		force_p[i] = make_double3(0, 0, 0);
		torque_p[i] = make_double3(0, 0, 0);
	}

	force_c[idx_c] += F_c;
	torque_c[idx_c] += T_c;
}

__global__ void solidParticleVelocityAngularVelocityIntegrateKernel(double3* velocity, double3* angularVelocity, 
    double3* force, double3* torque, double* radius, double* invMass, int* clumpID, 
	const double3 g,
	const double dt,
	const size_t numParticle)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numParticle) return;

	if (clumpID[idx_i] >= 0) return;
	double invM_i = invMass[idx_i];
	velocity[idx_i] += (force[idx_i] * invM_i + g * (invM_i > 0.0)) * dt;
	double rad_i = radius[idx_i];
	if (invM_i < 1.e-20 || rad_i < 1.e-20) return;
	double I_i = 0.4 * rad_i * rad_i / invM_i;
	angularVelocity[idx_i] += torque[idx_i] / I_i * dt;
}

__global__ void clumpVelocityAngularVelocityIntegrateKernel(double3* velocity_c, double3* angularVelocity_c, double3* position_c, 
    double3* force_c, double3* torque_c, double* invMass_c, quaternion* orientation, symMatrix* inverseInertiaTensor, 
	int* pebbleStartIndex, int* pebbleEndIndex,
	double3* velocity_p, double3* angularVelocity_p, double3* position_p, 
	const double3 g,
	const double dt, 
	const size_t numClumps)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClumps) return;

	velocity_c[idx_c] += (force_c[idx_c] * invMass_c[idx_c] + g * (invMass_c[idx_c] > 0.0)) * dt;
	double invM_c = invMass_c[idx_c];
	if (invM_c > 0.) angularVelocity_c[idx_c] += (rotateInverseInertiaTensor(orientation[idx_c], inverseInertiaTensor[idx_c]) * torque_c[idx_c]) * dt;
	double3 w_c = angularVelocity_c[idx_c];
	for (int i = pebbleStartIndex[idx_c]; i < pebbleEndIndex[idx_c]; i++)
	{
		double3 r_pc = position_p[i] - position_c[idx_c];
		velocity_p[i] = velocity_c[idx_c] + cross(w_c, r_pc);
		angularVelocity_p[i] = w_c;
	}
}

__global__ void positionIntegrateKernel(double3* position, double3* velocity, 
	const double dt,
	const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	position[idx_i] += dt * velocity[idx_i];
}

extern "C" void launchSolidParticleIntegrateBeforeContact(solidParticle& solidParticles, clump& clumps, const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream)
{
	size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);
	solidParticleVelocityAngularVelocityIntegrateKernel <<<grid, block, 0, stream>>> (solidParticles.velocity(), solidParticles.angularVelocity, 
	solidParticles.force, solidParticles.torque, solidParticles.radius, solidParticles.inverseMass, solidParticles.clumpID, 
	gravity, 0.5 * timeStep, solidParticles.deviceSize());

	computeGPUGridSizeBlockSize(grid, block, clumps.deviceSize(), maxThreadsPerBlock);
	clumpVelocityAngularVelocityIntegrateKernel <<<grid, block, 0, stream>>> (clumps.velocity(), clumps.angularVelocity, 
	clumps.position(), clumps.force, clumps.torque, clumps.inverseMass, clumps.orientation, clumps.inverseInertiaTensor, clumps.pebbleStartIndex, clumps.pebbleEndIndex, 
	solidParticles.velocity(), solidParticles.angularVelocity, solidParticles.position(), gravity, 0.5 * timeStep, clumps.deviceSize());

	positionIntegrateKernel <<<grid, block, 0, stream>>> (clumps.position(), clumps.velocity(), 0.5 * timeStep, clumps.deviceSize());

	computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);
	positionIntegrateKernel <<<grid, block, 0, stream>>> (solidParticles.position(), solidParticles.velocity(), 0.5 * timeStep, solidParticles.deviceSize());
}

extern "C" void launchSolidParticleIntegrateAfterContact(solidParticle& solidParticles, clump& clumps, const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream)
{
	size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);
	positionIntegrateKernel <<<grid, block, 0, stream>>> (solidParticles.position(), solidParticles.velocity(), 0.5 * timeStep, solidParticles.deviceSize());

	computeGPUGridSizeBlockSize(grid, block, clumps.deviceSize(), maxThreadsPerBlock);
	positionIntegrateKernel <<<grid, block, 0, stream>>> (clumps.position(), clumps.velocity(), 0.5 * timeStep, clumps.deviceSize());

	clumpVelocityAngularVelocityIntegrateKernel <<<grid, block, 0, stream>>> (clumps.velocity(), clumps.angularVelocity, 
	clumps.position(), clumps.force, clumps.torque, clumps.inverseMass, clumps.orientation, clumps.inverseInertiaTensor, clumps.pebbleStartIndex, clumps.pebbleEndIndex, 
	solidParticles.velocity(), solidParticles.angularVelocity, solidParticles.position(), gravity, 0.5 * timeStep, clumps.deviceSize());

	computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);
	solidParticleVelocityAngularVelocityIntegrateKernel <<<grid, block, 0, stream>>> (solidParticles.velocity(), solidParticles.angularVelocity, 
	solidParticles.force, solidParticles.torque, solidParticles.radius, solidParticles.inverseMass, solidParticles.clumpID, 
	gravity, 0.5 * timeStep, solidParticles.deviceSize());
}

extern "C" void launchSolidParticleInteractionCalculation(interactionSpringSystem& solidParticleInteractions, interactionBonded& bondedSolidParticleInteractions, solidParticle& solidParticles, clump& clumps,
solidContactModelParameter& contactModelParameters, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream)
{
	size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, solidParticleInteractions.getActiveNumber(), maxThreadsPerBlock);
	calSolidParticleContactForceTorqueKernel <<<grid, block, 0, stream>>> (
	solidParticleInteractions.current.force(),
	solidParticleInteractions.current.torque,
	solidParticleInteractions.current.slidingSpring,
	solidParticleInteractions.current.rollingSpring,
	solidParticleInteractions.current.torsionSpring,
	solidParticleInteractions.current.objectPointed(),
	solidParticleInteractions.current.objectPointing(),
	solidParticles.position(),
	solidParticles.velocity(),
	solidParticles.angularVelocity,
	solidParticles.radius,
	solidParticles.inverseMass,
	solidParticles.materialID,
	contactModelParameters.hertzian.E,
	contactModelParameters.hertzian.G,
	contactModelParameters.hertzian.res,
	contactModelParameters.hertzian.k_r_k_s,
	contactModelParameters.hertzian.k_t_k_s,
	contactModelParameters.hertzian.mu_s,
	contactModelParameters.hertzian.mu_r,
	contactModelParameters.hertzian.mu_t,
	contactModelParameters.linear.k_n,
	contactModelParameters.linear.k_s,
	contactModelParameters.linear.k_r,
	contactModelParameters.linear.k_t,
	contactModelParameters.linear.d_n,
	contactModelParameters.linear.d_s,
	contactModelParameters.linear.d_r,
	contactModelParameters.linear.d_t,
	contactModelParameters.linear.mu_s,
	contactModelParameters.linear.mu_r,
	contactModelParameters.linear.mu_t,
	contactModelParameters.getNumberOfMaterials(),
	contactModelParameters.size(),
	timeStep,
	solidParticleInteractions.getActiveNumber());

	computeGPUGridSizeBlockSize(grid, block, bondedSolidParticleInteractions.size(), maxThreadsPerBlock);
	calBondedForceTorqueKernel <<<grid, block, 0, stream>>> (
	bondedSolidParticleInteractions.normalForce,
	bondedSolidParticleInteractions.torsionTorque,
	bondedSolidParticleInteractions.shearForce,
	bondedSolidParticleInteractions.bendingTorque,
	bondedSolidParticleInteractions.contactNormal,
	bondedSolidParticleInteractions.isBonded,
	bondedSolidParticleInteractions.objectPointed,
	bondedSolidParticleInteractions.objectPointing,
	solidParticleInteractions.current.force(),
	solidParticleInteractions.current.torque,
	solidParticleInteractions.current.objectPointed(),
	solidParticleInteractions.current.objectPointing(),
	solidParticles.force,
	solidParticles.torque,
	solidParticles.position(),
	solidParticles.velocity(),
	solidParticles.angularVelocity,
	solidParticles.radius,
	solidParticles.inverseMass,
	solidParticles.materialID,
	solidParticles.neighbor.prefixSum,
	contactModelParameters.bonded.gamma,
	contactModelParameters.bonded.E,
	contactModelParameters.bonded.k_n_k_s,
	contactModelParameters.bonded.sigma_s,
	contactModelParameters.bonded.C,
	contactModelParameters.bonded.mu,
	contactModelParameters.getNumberOfMaterials(),
	contactModelParameters.size(),
	timeStep,
	bondedSolidParticleInteractions.size());

	computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);
	sumForceTorqueFromSolidParticleInteractionKernel <<<grid, block, 0, stream>>> (
	solidParticleInteractions.current.force(),
	solidParticleInteractions.current.torque,
	solidParticleInteractions.current.objectPointed(),
	solidParticleInteractions.current.objectPointing(),
	solidParticleInteractions.current.hash().index,
	solidParticles.force,
	solidParticles.torque,
	solidParticles.position(),
	solidParticles.radius,
	solidParticles.neighbor.prefixSum,
	solidParticles.interactionIndexRange.start,
	solidParticles.interactionIndexRange.end,
	solidParticles.deviceSize());

	computeGPUGridSizeBlockSize(grid, block, clumps.deviceSize(), maxThreadsPerBlock);
	sumClumpForceTorqueKernel <<<grid, block, 0, stream>>> (clumps.force, clumps.torque, 
	clumps.position(), clumps.pebbleStartIndex, clumps.pebbleEndIndex, 
	solidParticles.force, solidParticles.torque, solidParticles.position(), clumps.deviceSize());
}

extern "C" void launchSolidParticleInfiniteWallInteractionCalculation(interactionSpringSystem& solidParticleInfiniteWallInteractions, 
solidParticle& solidParticles, 
infiniteWall& infiniteWalls,
solidContactModelParameter& contactModelParameters, 
objectNeighborPrefix &solidParticleInfiniteWallNeighbor,
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, solidParticleInfiniteWallInteractions.getActiveNumber(), maxThreadsPerBlock);
	calSolidParticleInfiniteWallContactForceTorqueKernel <<<grid, block, 0, stream>>> (solidParticleInfiniteWallInteractions.current.force(),
	solidParticleInfiniteWallInteractions.current.torque,
	solidParticleInfiniteWallInteractions.current.slidingSpring,
	solidParticleInfiniteWallInteractions.current.rollingSpring,
	solidParticleInfiniteWallInteractions.current.torsionSpring,
	solidParticleInfiniteWallInteractions.current.objectPointed(),
	solidParticleInfiniteWallInteractions.current.objectPointing(),
	solidParticles.position(),
	solidParticles.velocity(),
	solidParticles.angularVelocity,
	solidParticles.radius,
	solidParticles.inverseMass,
	solidParticles.materialID,
	infiniteWalls.position(),
	infiniteWalls.velocity(),
	infiniteWalls.axis(),
	infiniteWalls.axisAngularVelocity(),
	infiniteWalls.radius(),
	infiniteWalls.materialID(),
	contactModelParameters.hertzian.E,
	contactModelParameters.hertzian.G,
	contactModelParameters.hertzian.res,
	contactModelParameters.hertzian.k_r_k_s,
	contactModelParameters.hertzian.k_t_k_s,
	contactModelParameters.hertzian.mu_s,
	contactModelParameters.hertzian.mu_r,
	contactModelParameters.hertzian.mu_t,
	contactModelParameters.linear.k_n,
	contactModelParameters.linear.k_s,
	contactModelParameters.linear.k_r,
	contactModelParameters.linear.k_t,
	contactModelParameters.linear.d_n,
	contactModelParameters.linear.d_s,
	contactModelParameters.linear.d_r,
	contactModelParameters.linear.d_t,
	contactModelParameters.linear.mu_s,
	contactModelParameters.linear.mu_r,
	contactModelParameters.linear.mu_t,
	contactModelParameters.getNumberOfMaterials(),
	contactModelParameters.size(),
	timeStep,
	solidParticleInfiniteWallInteractions.getActiveNumber());

	computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);
	sumForceTorqueFromSolidParticleInfiniteWallInteractionKernel <<<grid, block, 0, stream>>> (
	solidParticleInfiniteWallInteractions.current.force(),
	solidParticleInfiniteWallInteractions.current.torque,
	solidParticleInfiniteWallInteractions.current.objectPointing(),
	solidParticles.force,
	solidParticles.torque,
	solidParticles.position(),
	solidParticles.radius,
	solidParticleInfiniteWallNeighbor.prefixSum,
	infiniteWalls.position(),
	infiniteWalls.axis(),
	infiniteWalls.radius(),
	solidParticles.deviceSize());
}

extern "C" void launchInfiniteWallHalfIntegration(infiniteWall &infiniteWalls, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream)
{
	size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, infiniteWalls.deviceSize(), maxThreadsPerBlock);
	positionIntegrateKernel <<<grid, block, 0, stream>>> (infiniteWalls.position(), 
	infiniteWalls.velocity(),
	0.5 * timeStep, 
	infiniteWalls.deviceSize());
}