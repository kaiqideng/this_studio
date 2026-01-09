#include"ballMeshWallIntegration.h"

__global__ void calBallTriangleContactSpringContactPoint(double3* contactPoint,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
int* objectPointing,
int* cancelFlag, 
double3* position, 
const double* radius, 
int* interactionMapPrefixSumA,
const int* vertIndex0_t, 
const int* vertIndex1_t, 
const int* vertIndex2_t, 
double3* globalVertices,
const size_t numBalls)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBalls) return;

    int start = 0;
	if (idx_i > 0) start = interactionMapPrefixSumA[idx_i - 1];
	int end = interactionMapPrefixSumA[idx_i];
	for(int idx_c = start; idx_c < end; idx_c++)
	{
		cancelFlag[idx_c] = 0;
		
		const int idx_j = objectPointing[idx_c];

		const double rad_i = radius[idx_i];
		const double3 r_i = position[idx_i];

		const double3 p0 = globalVertices[vertIndex0_t[idx_j]];
		const double3 p1 = globalVertices[vertIndex1_t[idx_j]];
		const double3 p2 = globalVertices[vertIndex2_t[idx_j]];
		
		double3 r_c;
		SphereTriangleContactType type = classifySphereTriangleContact(r_i, rad_i,
									p0, p1, p2,
									r_c);

		contactPoint[idx_c] = r_c;

        if(type == SphereTriangleContactType::None) continue;

		if(type != SphereTriangleContactType::Face)
		{
			for(int idx_c1 = start; idx_c1 < end; idx_c1++)
			{
				if(idx_c1 == idx_c) continue;

				const int idx_j1 = objectPointing[idx_c1];
				const double3 p01 = globalVertices[vertIndex0_t[idx_j1]];
				const double3 p11 = globalVertices[vertIndex1_t[idx_j1]];
				const double3 p21 = globalVertices[vertIndex2_t[idx_j1]];

				double3 r_c1;
				SphereTriangleContactType type1 = classifySphereTriangleContact(r_i, rad_i,
									p01, p11, p21,
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

__global__ void calBallTriangleContactForceTorqueKernel(double3* contactForce, 
double3* contactTorque,
double3* contactPoint,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
int* objectPointing, 
int* cancelFlag,
double3* force,
double3* torque,
double3* position, 
double3* velocity, 
double3* angularVelocity, 
const double* radius, 
const double* inverseMass, 
const int* materialID, 
double3* position_w, 
double3* velocity_w, 
double3* angularVelocity_w, 
const int* materialID_w, 
const int* wallIndex_t, 
int* interactionMapPrefixSumA, 
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
const size_t numBalls)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBalls) return;

    int start = 0;
	if(idx_i > 0) start = interactionMapPrefixSumA[idx_i - 1];
	int end = interactionMapPrefixSumA[idx_i];
	for(int idx_c = start; idx_c < end; idx_c++)
	{
		if (cancelFlag[idx_c] == 1) continue;

		contactForce[idx_c] = make_double3(0, 0, 0);
	    contactTorque[idx_c] = make_double3(0, 0, 0);

		const double rad_i = radius[idx_i];
		const double3 r_i = position[idx_i];
		
		double3 r_c = contactPoint[idx_c];
		double3 n_ij = normalize(r_i - r_c);
		double delta = rad_i - length(r_i - r_c);

		const double m_ij = 1. / inverseMass[idx_i];
		const double rad_ij = rad_i;

		const int idx_j = objectPointing[idx_c];
        const size_t idx_w = wallIndex_t[idx_j];
		const double3 r_w = position_w[idx_w];

		const double3 v_i = velocity[idx_i];
		const double3 v_j = velocity_w[idx_w];
		const double3 w_i = angularVelocity[idx_i];
		const double3 w_j = angularVelocity_w[idx_w];
		const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_w));
		const double3 w_ij = w_i - w_j;

		double3 F_c = make_double3(0, 0, 0);
		double3 T_c = make_double3(0, 0, 0);
		double3 epsilon_s = slidingSpring[idx_c];
		double3 epsilon_r = rollingSpring[idx_c];
		double3 epsilon_t = torsionSpring[idx_c];

		const size_t param_ij = getContactParameterArraryIndex(materialID[idx_i], materialID_w[idx_w], 
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

		contactForce[idx_c] = F_c;
		contactTorque[idx_c] = T_c;
		slidingSpring[idx_c] = epsilon_s;
		rollingSpring[idx_c] = epsilon_r;
		torsionSpring[idx_c] = epsilon_t;

		force[idx_i] += F_c;
		torque[idx_i] += T_c + cross(r_c - r_i, F_c);
	}
}

__global__ void triangleGlobalVerticesIntegrationKernel(double3* globalVertices, 
const double3* localVertices,
const int* triangleIndex_v,
const int* numTrianglesPrefixSum_v,
const int* wallIndex_t,
quaternion* orientation_w,
double3* position_w,
const double dt,
const size_t numVertices)
{
	size_t idx_v = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_v >= numVertices) return;

	size_t idx_t = 0;
	if(idx_v > 0) idx_t = triangleIndex_v[numTrianglesPrefixSum_v[idx_v - 1]];
	size_t idx_w = wallIndex_t[idx_t];
	globalVertices[idx_v] = position_w[idx_w] + rotateVectorByQuaternion(orientation_w[idx_w], localVertices[idx_v]);
}

extern "C" void launchBallMeshWallInteractionCalculation(solidInteraction &ballTriangleInteractions,
ball &balls,
meshWall &meshWalls,
contactModelParameters &contactModelParams,
interactionMap &ballTriangleInteractionMap,
const double timeStep,
const size_t maxThreadsPerBlock,
cudaStream_t stream)
{
	size_t gridDim = 1, blockDim = 1;
	if (setGPUGridBlockDim(gridDim, blockDim, balls.deviceSize(), maxThreadsPerBlock))
	{
		calBallTriangleContactSpringContactPoint <<<gridDim, blockDim, 0, stream>>> (ballTriangleInteractions.contactPoint(),
		ballTriangleInteractions.slidingSpring(),
		ballTriangleInteractions.rollingSpring(),
		ballTriangleInteractions.torsionSpring(),
		ballTriangleInteractions.objectPointing(),
		ballTriangleInteractions.cancelFlag(),
		balls.position(),
		balls.radius(),
		ballTriangleInteractionMap.prefixSumA(),
		meshWalls.triangles().index0(),
		meshWalls.triangles().index1(),
		meshWalls.triangles().index2(),
		meshWalls.globalVertices(),
		balls.deviceSize());

		calBallTriangleContactForceTorqueKernel <<<gridDim, blockDim, 0, stream>>> (ballTriangleInteractions.force(),
		ballTriangleInteractions.torque(),
		ballTriangleInteractions.contactPoint(),
		ballTriangleInteractions.slidingSpring(),
		ballTriangleInteractions.rollingSpring(),
		ballTriangleInteractions.torsionSpring(),
		ballTriangleInteractions.objectPointing(),
		ballTriangleInteractions.cancelFlag(),
		balls.force(),
		balls.torque(),
		balls.position(),
		balls.velocity(),
		balls.angularVelocity(),
		balls.radius(),
		balls.inverseMass(),
		balls.materialID(),
		meshWalls.position(),
		meshWalls.velocity(),
		meshWalls.angularVelocity(),
		meshWalls.materialID(),
		meshWalls.triangles().wallIndex(),
		ballTriangleInteractionMap.prefixSumA(),
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
		balls.deviceSize());
	}
}

extern "C" void launchMeshWallIntegration(meshWall &meshWalls, 
const double timeStep,
const size_t maxThreadsPerBlock,
cudaStream_t stream)
{
	size_t gridDim = 1, blockDim = 1;
	if (setGPUGridBlockDim(gridDim, blockDim, meshWalls.deviceSize(), maxThreadsPerBlock))
	{
		orientationIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (meshWalls.orientation(), 
		meshWalls.angularVelocity(),
		timeStep,
		meshWalls.deviceSize());

		positionIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (meshWalls.position(),
		meshWalls.velocity(),
		timeStep,
		meshWalls.deviceSize());

		if (setGPUGridBlockDim(gridDim, blockDim, meshWalls.vertices().deviceSize(), maxThreadsPerBlock))
		{
			triangleGlobalVerticesIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (meshWalls.globalVertices(),
			meshWalls.vertices().localPosition(),
			meshWalls.vertices().triangleIndex(),
			meshWalls.vertices().trianglesPrefixSum(),
			meshWalls.triangles().wallIndex(),
			meshWalls.orientation(),
			meshWalls.position(),
			timeStep,
			meshWalls.vertices().deviceSize());
		}
	}
}