#include "contactKernel.h"
#include "contactParameters.h"
#include "myUtility/myVec.h"

__constant__ ContactParamsDevice contactPara;

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

__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v)
{
    atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}

__global__ void updateBallContactKernel(double3* contactPoint,
double3* contactNormal,
double* overlap,
const int* objectPointed, 
const int* objectPointing, 
const double3* position, 
const double* radius,
const size_t numInteractions)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteractions) return;

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];

    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
    const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];

    const double3 n_ij = normalize(r_i - r_j);
    const double delta = rad_i + rad_j - length(r_i - r_j);
	const double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;

    contactPoint[idx] = r_c;
    contactNormal[idx] = n_ij;
    overlap[idx] = delta;
}

__global__ void calBallContactForceTorqueKernel(double3* contactForce, 
double3* contactTorque,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
const double3* contactPoint,
const double3* contactNormal,
const double* overlap,
const int* objectPointed, 
const int* objectPointing, 
const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const double* inverseMass,
const int* materialID,
const double dt,
const size_t numInteractions)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteractions) return;

	contactForce[idx] = make_double3(0, 0, 0);
	contactTorque[idx] = make_double3(0, 0, 0);

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];

    const double3 r_c = contactPoint[idx];
    const double3 n_ij = contactNormal[idx];
	const double delta = overlap[idx];
	
    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];
	
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

    int ip = contactPairIndex(materialID[idx_i], materialID[idx_j], contactPara.nMaterials, contactPara.cap);
    const double k_n = getLinearParam(ip, L_KN);
    if (k_n > 1.e-20)
    {
        const double k_s = getLinearParam(ip, L_KS);
        const double k_r = getLinearParam(ip, L_KR);
        const double k_t = getLinearParam(ip, L_KT);
        const double d_n = getLinearParam(ip, L_DN);
        const double d_s = getLinearParam(ip, L_DS);
        const double d_r = getLinearParam(ip, L_DR);
        const double d_t = getLinearParam(ip, L_DT);
        const double mu_s = getLinearParam(ip, L_MU_S);
        const double mu_r = getLinearParam(ip, L_MU_R);
        const double mu_t = getLinearParam(ip, L_MU_T);

        LinearContact(F_c, 
        T_c, 
        epsilon_s, 
        epsilon_r, 
        epsilon_t,
		v_c_ij,
		w_ij,
		n_ij,
		delta,
		m_ij,
		rad_ij,
		dt,
		k_n,
		k_s,
		k_r,
		k_t,
		d_n,
		d_s,
		d_r,
		d_t,
		mu_s,
		mu_r,
		mu_t);
    }
    else 
    {
        const double logR = log(getHertzianParam(ip, H_RES));
		const double D = -logR / sqrt(logR * logR + pi() * pi());
        const double E = getHertzianParam(ip, H_E_STAR);
        const double G = getHertzianParam(ip, H_G_STAR);
        const double k_r_k_s = getHertzianParam(ip, H_KRKS);
        const double k_t_k_s = getHertzianParam(ip, H_KTKS);
        const double mu_s = getLinearParam(ip, H_MU_S);
        const double mu_r = getLinearParam(ip, H_MU_R);
        const double mu_t = getLinearParam(ip, H_MU_T);

        HertzianMindlinContact(F_c, 
        T_c, 
        epsilon_s, 
        epsilon_r, 
        epsilon_t,
		v_c_ij,
		w_ij,
		n_ij,
		delta,
		m_ij,
		rad_ij,
		dt,
		D,
		E,
		G,
		k_r_k_s,
		k_t_k_s,
		mu_s,
		mu_r,
		mu_t);
    }

    contactForce[idx] = F_c;
	contactTorque[idx] = T_c;
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void updateBallTriangleContact(double3* contactPoint,
double3* contactNormal,
double* overlap,
int* cancelFlag, 
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 

const int* objectPointing,

const double3* position, 
const double* radius, 
const int* neighborPrefixSum,

const int* index0_t, 
const int* index1_t, 
const int* index2_t, 

const double3* globalPosition_v,

const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

    int start = 0;
	if (idx_i > 0) start = neighborPrefixSum[idx_i - 1];
	int end = neighborPrefixSum[idx_i];
	for(int idx_c = start; idx_c < end; idx_c++)
	{
		cancelFlag[idx_c] = 0;
		
		const int idx_j = objectPointing[idx_c];

		const double rad_i = radius[idx_i];
		const double3 r_i = position[idx_i];

		const double3 p0 = globalPosition_v[index0_t[idx_j]];
		const double3 p1 = globalPosition_v[index1_t[idx_j]];
		const double3 p2 = globalPosition_v[index2_t[idx_j]];
		
		double3 r_c;
		SphereTriangleContactType type = classifySphereTriangleContact(r_i, 
		rad_i,
		p0, 
		p1, 
		p2,
		r_c);

		contactPoint[idx_c] = r_c;
        contactNormal[idx_c] = normalize(r_i - r_c);
        overlap[idx_c] = rad_i - length(r_i - r_c);

        if (type == SphereTriangleContactType::None) 
        {
            cancelFlag[idx_c] = 1;
            continue;
        }

		if (type != SphereTriangleContactType::Face)
		{
			for(int idx_c1 = start; idx_c1 < end; idx_c1++)
			{
				if(idx_c1 == idx_c) continue;

				const int idx_j1 = objectPointing[idx_c1];
				const double3 p01 = globalPosition_v[index0_t[idx_j1]];
				const double3 p11 = globalPosition_v[index1_t[idx_j1]];
				const double3 p21 = globalPosition_v[index2_t[idx_j1]];

				double3 r_c1;
				SphereTriangleContactType type1 = classifySphereTriangleContact(r_i, 
				rad_i,
				p01, 
				p11, 
				p21,
				r_c1);
				
				if(type1 == SphereTriangleContactType::None) continue;
				else if(type1 == SphereTriangleContactType::Face)
				{
					// Find sharing face
					if(lengthSquared(cross(r_c - p01, p11 - p01)) < 1.e-20) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if(lengthSquared(cross(r_c - p11, p21 - p11)) < 1.e-20)
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if(lengthSquared(cross(r_c - p01, p21 - p01)) < 1.e-20) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
				}
				else if(type1 == SphereTriangleContactType::Edge)
				{
					if(type == type1)
					{
						if(idx_c1 < idx_c)
						{
							if(lengthSquared(r_c - r_c1) < 1.e-20)
							{
								slidingSpring[idx_c] = slidingSpring[idx_c1];
								rollingSpring[idx_c] = rollingSpring[idx_c1];
								torsionSpring[idx_c] = torsionSpring[idx_c1];
								cancelFlag[idx_c] = 1;
								break;
							}
						}
					}
					else 
					{
						if(lengthSquared(r_c - r_c1) < 1.e-20)
						{
							slidingSpring[idx_c] = slidingSpring[idx_c1];
							rollingSpring[idx_c] = rollingSpring[idx_c1];
							torsionSpring[idx_c] = torsionSpring[idx_c1];
							cancelFlag[idx_c] = 1;
							break;
						}
					}
				}
				else
				{
					if(type == type1)
					{
						if(idx_c1 < idx_c)
						{
							if(lengthSquared(r_c - r_c1) < 1.e-20)
							{
								slidingSpring[idx_c] = slidingSpring[idx_c1];
								rollingSpring[idx_c] = rollingSpring[idx_c1];
								torsionSpring[idx_c] = torsionSpring[idx_c1];
								cancelFlag[idx_c] = 1;
								break;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void calBallWallContactForceTorqueKernel(double3* contactForce, 
double3* contactTorque,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
const double3* contactPoint,
const double3* contactNormal,
const double* overlap,
const int* objectPointed, 
const int* objectPointing, 
const int* cancelFlag, 
const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const double* inverseMass,
const int* materialID,
const double3* position_w, 
const double3* velocity_w, 
const double3* angularVelocity_w, 
const int* materialID_w,
const double dt,
const size_t numInteractions)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteractions) return;

	contactForce[idx] = make_double3(0, 0, 0);
	contactTorque[idx] = make_double3(0, 0, 0);
    if (cancelFlag[idx] == 1) return;

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];
    if (inverseMass[idx_i] < 1.e-20) return;

	const double3 r_c = contactPoint[idx];
    const double3 n_ij = contactNormal[idx];
    const double delta = overlap[idx];

    const double rad_ij = radius[idx_i];
	const double m_ij = 1. / (inverseMass[idx_i]);

	const double3 r_i = position[idx_i];
	const double3 r_j = position_w[idx_j];
	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity_w[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity_w[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;

	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	double3 epsilon_s = slidingSpring[idx];
	double3 epsilon_r = rollingSpring[idx];
	double3 epsilon_t = torsionSpring[idx];

    int ip = contactPairIndex(materialID[idx_i], materialID[idx_j], contactPara.nMaterials, contactPara.cap);
    const double k_n = getLinearParam(ip, L_KN);
    if (k_n > 1.e-20)
    {
        const double k_s = getLinearParam(ip, L_KS);
        const double k_r = getLinearParam(ip, L_KR);
        const double k_t = getLinearParam(ip, L_KT);
        const double d_n = getLinearParam(ip, L_DN);
        const double d_s = getLinearParam(ip, L_DS);
        const double d_r = getLinearParam(ip, L_DR);
        const double d_t = getLinearParam(ip, L_DT);
        const double mu_s = getLinearParam(ip, L_MU_S);
        const double mu_r = getLinearParam(ip, L_MU_R);
        const double mu_t = getLinearParam(ip, L_MU_T);

        LinearContact(F_c, 
        T_c, 
        epsilon_s, 
        epsilon_r, 
        epsilon_t,
		v_c_ij,
		w_ij,
		n_ij,
		delta,
		m_ij,
		rad_ij,
		dt,
		k_n,
		k_s,
		k_r,
		k_t,
		d_n,
		d_s,
		d_r,
		d_t,
		mu_s,
		mu_r,
		mu_t);
    }
    else 
    {
        const double logR = log(getHertzianParam(ip, H_RES));
		const double D = -logR / sqrt(logR * logR + pi() * pi());
        const double E = getHertzianParam(ip, H_E_STAR);
        const double G = getHertzianParam(ip, H_G_STAR);
        const double k_r_k_s = getHertzianParam(ip, H_KRKS);
        const double k_t_k_s = getHertzianParam(ip, H_KTKS);
        const double mu_s = getLinearParam(ip, H_MU_S);
        const double mu_r = getLinearParam(ip, H_MU_R);
        const double mu_t = getLinearParam(ip, H_MU_T);

        HertzianMindlinContact(F_c, 
        T_c, 
        epsilon_s, 
        epsilon_r, 
        epsilon_t,
		v_c_ij,
		w_ij,
		n_ij,
		delta,
		m_ij,
		rad_ij,
		dt,
		D,
		E,
		G,
		k_r_k_s,
		k_t_k_s,
		mu_s,
		mu_r,
		mu_t);
    }

    contactForce[idx] = F_c;
	contactTorque[idx] = T_c;
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void calBondedForceTorqueKernel(int* isBonded, 
double* normalForce, 
double* torsionTorque, 
double3* shearForce, 
double3* bendingTorque,
double3* bondPoint,
double3* bondNormal,

double3* contactForce, 
double3* contactTorque,

double3* force, 
double3* torque, 

const int* objectPointed_b, 
const int* objectPointing_b,

const double3* contactPoint,
const double3* contactNormal,
const int* objectPointed, 
const int* objectPointing,

const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const int* materialID, 
const int* neighborPrefixSum,

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
		return;
	}

	const int idx_i = objectPointed_b[idx];
	const int idx_j = objectPointing_b[idx];

    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];

    const double3 n_ij0 = bondNormal[idx];
	double3 n_ij = normalize(r_i - r_j);
    const double delta = rad_i + rad_j - length(r_i - r_j);
	double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
    bondPoint[idx] = r_c;
    bondNormal[idx] = n_ij;

	bool flag = false;
	int idx_c = 0;
	const int neighborStart_i = idx_i > 0 ? neighborPrefixSum[idx_i - 1] : 0;
	const int neighborEnd_i = neighborPrefixSum[idx_i];
	for (int k = neighborStart_i; k < neighborEnd_i; k++)
	{
		if (objectPointing[k] == idx_j)
		{
			flag = true;
			idx_c = k;
            r_c = contactPoint[idx_c];
            n_ij = contactNormal[idx_c];
            bondPoint[idx] = r_c;
            bondNormal[idx] = n_ij;
			break;
		}
	}
	
	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));

    int ip = contactPairIndex(materialID[idx_i], materialID[idx_j], contactPara.nMaterials, contactPara.cap);
	const double gamma = getBondedParam(ip, B_GAMMA);
    const double E_b = getBondedParam(ip, B_EB);
    const double k_n_k_s = getBondedParam(ip, B_KNKS);
    const double sigma_s = getBondedParam(ip, B_SIGMA_S);
    const double C = getBondedParam(ip, B_C);
    const double mu = getBondedParam(ip, B_MU);

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
	gamma,
	E_b,
	k_n_k_s,
	sigma_s,
	C,
	mu);

	normalForce[idx] = F_n;
	shearForce[idx] = F_s;
	torsionTorque[idx] = T_t;
	bendingTorque[idx] = T_b;

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

__global__ void sumObjectPointedForceTorqueFromInteractionKernel(double3* force, 
double3* torque, 
const double3* position, 
const int* prefixSumA,
const double3* contactForce, 
const double3* contactTorque, 
const double3* contactPoint,
const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0, 0, 0);
	double3 T_i = make_double3(0, 0, 0);
	for (int k = idx_i > 0 ? prefixSumA[idx_i - 1] : 0; k < prefixSumA[idx_i]; k++)
	{
		double3 r_c = contactPoint[k];
		F_i += contactForce[k];
		T_i += contactTorque[k] + cross(r_c - r_i, contactForce[k]);
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}

__global__ void sumObjectPointingForceTorqueFromInteractionKernel(double3* force, 
double3* torque, 
const double3* position, 
const int* interactionStartB, 
const int* interactionEndB,
const double3* contactForce, 
const double3* contactTorque, 
const double3* contactPoint,
const int* objectPointing,
const int* interactionHashIndex,
const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0, 0, 0);
	double3 T_i = make_double3(0, 0, 0);
	if (interactionStartB[idx_i] != 0xFF)
	{
		for (int k = interactionStartB[idx_i]; k < interactionEndB[idx_i]; k++)
		{
			int k1 = interactionHashIndex[k];
			double3 r_c = contactPoint[k1];
			F_i -= contactForce[k1];
			T_i -= contactTorque[k1];
			T_i -= cross(r_c - r_i, contactForce[k1]);
		}
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}