#include "contactKernel.h"
#include "contactParameters.h"

__constant__ ContactParamsDevice contactPara;

void contactModelParameters::buildFromTables(const std::vector<HertzianRow>& hertzianTable,
const std::vector<LinearRow>& linearTable,
const std::vector<BondedRow>& bondedTable,
cudaStream_t stream)
{
    const int nMat = inferNumberOfMaterials_(hertzianTable, linearTable, bondedTable);
    if (nMat <= 0) return;

    numberOfMaterials = static_cast<std::size_t>(nMat);
    pairTableSize = computePairTableSize_(nMat);
    if (pairTableSize == 0) return;

    std::vector<double> H(static_cast<std::size_t>(H_COUNT) * pairTableSize, 0.0);
    std::vector<double> L(static_cast<std::size_t>(L_COUNT) * pairTableSize, 0.0);
    std::vector<double> B(static_cast<std::size_t>(B_COUNT) * pairTableSize, 0.0);

    for (std::size_t idx = 0; idx < pairTableSize; ++idx)
    {
        setPacked_(H, pairTableSize, H_RES, idx, 1.0);
    }

    for (std::size_t idx = 0; idx < pairTableSize; ++idx)
    {
        setPacked_(B, pairTableSize, B_GAMMA, idx, 1.0);
        setPacked_(B, pairTableSize, B_KNKS,  idx, 1.0);
    }

    auto pairIdx = [&](int a, int b) -> std::size_t
    {
        return static_cast<std::size_t>(contactPairParameterIndex(a, b, nMat, static_cast<int>(pairTableSize)));
    };

    for (const auto& row : hertzianTable)
    {
        const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

        setPacked_(H, pairTableSize, H_E_STAR, idx, row.effectiveYoungsModulus);
        setPacked_(H, pairTableSize, H_G_STAR, idx, row.effectiveShearModulus);
        setPacked_(H, pairTableSize, H_RES,    idx, row.restitutionCoefficient);
        setPacked_(H, pairTableSize, H_KRKS,   idx, row.rollingStiffnessToShearStiffnessRatio);
        setPacked_(H, pairTableSize, H_KTKS,   idx, row.torsionStiffnessToShearStiffnessRatio);
        setPacked_(H, pairTableSize, H_MU_S,   idx, row.slidingFrictionCoefficient);
        setPacked_(H, pairTableSize, H_MU_R,   idx, row.rollingFrictionCoefficient);
        setPacked_(H, pairTableSize, H_MU_T,   idx, row.torsionFrictionCoefficient);
    }

    for (const auto& row : linearTable)
    {
        const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

        setPacked_(L, pairTableSize, L_KN,   idx, row.normalStiffness);
        setPacked_(L, pairTableSize, L_KS,   idx, row.slidingStiffness);
        setPacked_(L, pairTableSize, L_KR,   idx, row.rollingStiffness);
        setPacked_(L, pairTableSize, L_KT,   idx, row.torsionStiffness);

        setPacked_(L, pairTableSize, L_DN,   idx, row.normalDampingCoefficient);
        setPacked_(L, pairTableSize, L_DS,   idx, row.slidingDampingCoefficient);
        setPacked_(L, pairTableSize, L_DR,   idx, row.rollingDampingCoefficient);
        setPacked_(L, pairTableSize, L_DT,   idx, row.torsionDampingCoefficient);

        setPacked_(L, pairTableSize, L_MU_S, idx, row.slidingFrictionCoefficient);
        setPacked_(L, pairTableSize, L_MU_R, idx, row.rollingFrictionCoefficient);
        setPacked_(L, pairTableSize, L_MU_T, idx, row.torsionFrictionCoefficient);
    }

    for (const auto& row : bondedTable)
    {
        const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

        setPacked_(B, pairTableSize, B_GAMMA,   idx, row.bondRadiusMultiplier);
        setPacked_(B, pairTableSize, B_EB,      idx, row.bondYoungsModulus);
        setPacked_(B, pairTableSize, B_KNKS,    idx, row.normalToShearStiffnessRatio);
        setPacked_(B, pairTableSize, B_SIGMA_S, idx, row.tensileStrength);
        setPacked_(B, pairTableSize, B_C,       idx, row.cohesion);
        setPacked_(B, pairTableSize, B_MU,      idx, row.frictionCoefficient);
    }

    hertzianPacked_.setHost(H);
    linearPacked_.setHost(L);
    bondedPacked_.setHost(B);

    hertzianPacked_.copyHostToDevice(stream);
    linearPacked_.copyHostToDevice(stream);
    bondedPacked_.copyHostToDevice(stream);

    ContactParamsDevice dev;
    dev.nMaterials = static_cast<int>(numberOfMaterials);
    dev.cap = static_cast<int>(pairTableSize);
    dev.hertzian = hertzianPacked_.d_ptr;
    dev.linear   = linearPacked_.d_ptr;
    dev.bonded   = bondedPacked_.d_ptr;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(contactPara,
                                       &dev,
                                       sizeof(ContactParamsDevice),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       stream));
}

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
const size_t numInteraction)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteraction) return;

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];

    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
    const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];

    const double3 r_ij = r_i - r_j;
    const double3 n_ij = normalize(r_ij);
    const double delta = rad_i + rad_j - length(r_ij);
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
const size_t numInteraction)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteraction) return;

	contactForce[idx] = make_double3(0., 0., 0.);
	contactTorque[idx] = make_double3(0., 0., 0.);

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
	const double m_ij = 1. / (inverseMass[idx_i] + inverseMass[idx_j]); // exclude (inverseMass[idx_i] == 0 && inverseMass[idx_j] == 0) 

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

    int ip = contactPairParameterIndex(materialID[idx_i], materialID[idx_j], contactPara.nMaterials, contactPara.cap);
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

        LinearContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
		v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt, 
		k_n, k_s, k_r, k_t, d_n, d_s, d_r, d_t, mu_s, mu_r, mu_t);
    }
    else 
    {
        const double logR = log(getHertzianParam(ip, H_RES));
		const double D = -logR / sqrt(logR * logR + pi() * pi());
        const double E = getHertzianParam(ip, H_E_STAR);
        const double G = getHertzianParam(ip, H_G_STAR);
        const double k_r_k_s = getHertzianParam(ip, H_KRKS);
        const double k_t_k_s = getHertzianParam(ip, H_KTKS);
        const double mu_s = getHertzianParam(ip, H_MU_S);
        const double mu_r = getHertzianParam(ip, H_MU_R);
        const double mu_t = getHertzianParam(ip, H_MU_T);

        HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
		v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt,
		D, E, G, k_r_k_s, k_t_k_s, mu_s, mu_r, mu_t);
    }

    contactForce[idx] = F_c;
	contactTorque[idx] = T_c;
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void updateBallTriangleContact(double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* cancelFlag, 
 
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
	for (int idx_c = start; idx_c < end; idx_c++)
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
			slidingSpring[idx_c] = make_double3(0., 0., 0.);
			rollingSpring[idx_c] = make_double3(0., 0., 0.);
			torsionSpring[idx_c] = make_double3(0., 0., 0.);
			cancelFlag[idx_c] = 1;
            continue;
        }

		if (type != SphereTriangleContactType::Face)
		{
			for (int idx_c1 = start; idx_c1 < end; idx_c1++)
			{
				if (idx_c1 == idx_c) continue;

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
				
				if (type1 == SphereTriangleContactType::None) continue;
				else if (type1 == SphereTriangleContactType::Face)
				{
					// Find sharing face
					if (lengthSquared(cross(r_c - p01, p11 - p01)) < 1.e-20) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if (lengthSquared(cross(r_c - p11, p21 - p11)) < 1.e-20)
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if (lengthSquared(cross(r_c - p01, p21 - p01)) < 1.e-20) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
				}
				else if (type1 == SphereTriangleContactType::Edge)
				{
					if (type == type1)
					{
						if (idx_c1 < idx_c)
						{
							if (lengthSquared(r_c - r_c1) < 1.e-20)
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
						if (lengthSquared(r_c - r_c1) < 1.e-20)
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
					if (type == type1)
					{
						if (idx_c1 < idx_c)
						{
							if (lengthSquared(r_c - r_c1) < 1.e-20)
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

__global__ void calBallWallContactForceTorqueKernel(double3* force,
double3* torque,
double3* contactForce, 
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
const int* neighborPrefixSum,
const double3* position_w, 
const double3* velocity_w, 
const double3* angularVelocity_w, 
const int* materialID_w,
const int* wallIndex_tri,
const double dt,
const size_t numBall)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;
	if (inverseMass[idx_i] < 1.e-20) return;

	const double3 r_i = position[idx_i];
	const double3 v_i = velocity[idx_i];
	const double3 w_i = angularVelocity[idx_i];
	const int materialID_i = materialID[idx_i];

	const double rad_ij = radius[idx_i];
	const double m_ij = 1. / inverseMass[idx_i];

	int start = 0;
	if (idx_i > 0) start = neighborPrefixSum[idx_i - 1];
	int end = neighborPrefixSum[idx_i];
	for (int idx_c = start; idx_c < end; idx_c++)
	{
		contactForce[idx_c] = make_double3(0, 0, 0);
		contactTorque[idx_c] = make_double3(0, 0, 0);
		if (cancelFlag[idx_c] == 1) continue;

		const int idx_j = objectPointing[idx_c];
		const int idx_w = wallIndex_tri[idx_j];

		const double3 r_c = contactPoint[idx_c];
		const double3 n_ij = contactNormal[idx_c];
		const double delta = overlap[idx_c];

		const double3 r_j = position_w[idx_w];
		const double3 v_j = velocity_w[idx_w];
		const double3 w_j = angularVelocity_w[idx_w];
		const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
		const double3 w_ij = w_i - w_j;

		double3 F_c = make_double3(0, 0, 0);
		double3 T_c = make_double3(0, 0, 0);
		double3 epsilon_s = slidingSpring[idx_c];
		double3 epsilon_r = rollingSpring[idx_c];
		double3 epsilon_t = torsionSpring[idx_c];

		int ip = contactPairParameterIndex(materialID_i, materialID_w[idx_w], contactPara.nMaterials, contactPara.cap);
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

			LinearContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt, 
			k_n, k_s, k_r, k_t, d_n, d_s, d_r, d_t, mu_s, mu_r, mu_t);
		}
		else 
		{
			const double logR = log(getHertzianParam(ip, H_RES));
			const double D = -logR / sqrt(logR * logR + pi() * pi());
			const double E = getHertzianParam(ip, H_E_STAR);
			const double G = getHertzianParam(ip, H_G_STAR);
			const double k_r_k_s = getHertzianParam(ip, H_KRKS);
			const double k_t_k_s = getHertzianParam(ip, H_KTKS);
			const double mu_s = getHertzianParam(ip, H_MU_S);
			const double mu_r = getHertzianParam(ip, H_MU_R);
			const double mu_t = getHertzianParam(ip, H_MU_T);

			HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt,
			D, E, G, k_r_k_s, k_t_k_s, mu_s, mu_r, mu_t);
		}

		contactForce[idx_c] = F_c;
		contactTorque[idx_c] = T_c;
		slidingSpring[idx_c] = epsilon_s;
		rollingSpring[idx_c] = epsilon_r;
		torsionSpring[idx_c] = epsilon_t;

		force[idx_i] += F_c;
		torque[idx_i] += T_c + cross(r_c - r_i, F_c);
	}
}

__global__ void calBondedForceTorqueKernel(double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
int* isBonded, 

double3* contactForce, 
double3* contactTorque,

double3* force, 
double3* torque, 

const int* objectPointed_b, 
const int* objectPointing_b,

const double3* contactPoint,
const double3* contactNormal,
const int* objectPointing,

const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const int* materialID, 
const int* neighborPrefixSum,

const double dt,
const size_t numBondedInteraction)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numBondedInteraction) return;

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

    int ip = contactPairParameterIndex(materialID[idx_i], materialID[idx_j], contactPara.nMaterials, contactPara.cap);
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
	n_ij0, n_ij, v_c_ij, w_i, w_j, rad_i, rad_j, dt,
	gamma, E_b, k_n_k_s, sigma_s, C, mu);

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
const int* neighborPrefixSum,
const double3* contactForce, 
const double3* contactTorque, 
const double3* contactPoint,
const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0., 0., 0.);
	double3 T_i = make_double3(0., 0., 0.);
	for (int k = idx_i > 0 ? neighborPrefixSum[idx_i - 1] : 0; k < neighborPrefixSum[idx_i]; k++)
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
const int* interactionStart, 
const int* interactionEnd,
const double3* contactForce, 
const double3* contactTorque, 
const double3* contactPoint,
const int* interactionHashIndex,
const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0., 0., 0.);
	double3 T_i = make_double3(0., 0., 0.);
	if (interactionStart[idx_i] != -1)
	{
		for (int k = interactionStart[idx_i]; k < interactionEnd[idx_i]; k++)
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

extern "C" void luanchCalculateBallContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double* radius,
double* inverseMass,
int* materialID,

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,

const double timeStep,

const size_t numInteraction,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	updateBallContactKernel <<<gridD, blockD, 0, stream>>> (contactPoint, 
	contactNormal, 
	overlap, 
	objectPointed, 
	objectPointing, 
	position, 
	radius, 
	numInteraction);

	calBallContactForceTorqueKernel <<<gridD, blockD, 0, stream>>> (contactForce,
	contactTorque,
	slidingSpring,
	rollingSpring,
	torsionSpring,
	contactPoint,
	contactNormal,
	overlap,
	objectPointed,
	objectPointing,
	position,
	velocity,
	angularVelocity,
	radius,
	inverseMass,
	materialID,
	timeStep,
	numInteraction);
}

extern "C" void luanchCalculateBondedForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque,
double* radius,
int* materialID,
int* neighborPrefixSum,

double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
int* objectPointing, 

double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
int* isBonded, 
int* objectPointed_b, 
int* objectPointing_b,

const double timeStep,

const size_t numBondedInteraction,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	calBondedForceTorqueKernel <<<gridD, blockD, 0, stream>>> (bondPoint, 
	bondNormal, 
	shearForce, 
	bendingTorque, 
	normalForce, 
	torsionTorque, 
	isBonded, 
	contactForce, 
	contactTorque, 
	force, 
	torque, 
	objectPointed_b, 
	objectPointing_b, 
	contactPoint, 
	contactNormal, 
	objectPointing, 
	position, 
	velocity, 
	angularVelocity, 
	radius, 
	materialID, 
	neighborPrefixSum, 
	timeStep, 
	numBondedInteraction);
}

extern "C" void luanchSumBallContactForceTorque(double3* position, 
double3* force, 
double3* torque,
int* neighborPrefixSum,
int* interactionStart, 
int* interactionEnd,

double3* contactForce,
double3* contactTorque,
double3* contactPoint,
int* interactionHashIndex,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	sumObjectPointedForceTorqueFromInteractionKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque, 
	position, 
	neighborPrefixSum, 
	contactForce, 
	contactTorque, 
	contactPoint, 
	numBall);

	sumObjectPointingForceTorqueFromInteractionKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque, 
	position, 
	interactionStart, 
	interactionEnd, 
	contactForce, 
	contactTorque, 
	contactPoint, 
	interactionHashIndex, 
	numBall);
}

void luanchCalculateBallWallContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque,
double* radius,
double* inverseMass,
int* materialID,
int* neighborPrefixSum,

double3* position_w, 
double3* velocity_w, 
double3* angularVelocity_w, 
int* materialID_w,

int* index0_t, 
int* index1_t, 
int* index2_t, 
int* wallIndex_tri,

double3* globalPosition_v, 

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring,
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,
int* cancelFlag,

const double timeStep,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	updateBallTriangleContact <<<gridD, blockD, 0, stream>>> (slidingSpring, 
	rollingSpring, 
	torsionSpring, 
	contactPoint, 
	contactNormal, 
	overlap, 
	cancelFlag, 
	objectPointing, 
	position, 
	radius, 
	neighborPrefixSum, 
	index0_t, 
	index1_t, 
	index2_t, 
	globalPosition_v, 
	numBall);

	calBallWallContactForceTorqueKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque, 
	contactForce, 
	contactTorque, 
	slidingSpring, 
	rollingSpring, 
	torsionSpring, 
	contactPoint, 
	contactNormal, 
	overlap, 
	objectPointed,
	objectPointing, 
	cancelFlag, 
	position, 
	velocity, 
	angularVelocity, 
	radius, 
	inverseMass, 
	materialID, 
    neighborPrefixSum,
	position_w, 
	velocity_w, 
	angularVelocity_w, 
	materialID_w, 
	wallIndex_tri,
	timeStep, 
	numBall);
}
