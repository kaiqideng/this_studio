#include"ballMeshWallIntegration.h"

__global__ void calBallTriangleContactForceTorqueKernel(double3* contactForce, 
double3* contactTorque,
double3* contactPoint,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
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
double3* position_w, 
double3* velocity_w, 
double3* angularVelocity_w, 
const int* materialID_w,
const int* wallIndex_t, 
const int* vertIndex0_t, 
const int* vertIndex1_t, 
const int* vertIndex2_t, 
double3* globalVertices,
double* hertzianE, 
double* hertzianG, 
double* hertzianRes, 
double* hertzianK_r_k_s, 
double* hertzianK_t_k_s, 
double* hertzianMu_s, 
double* hertzianMu_r, 
double* hertzianMu_t, 
double* linearK_n, 
double* linearK_s, 
double* linearK_r, 
double* linearK_t, 
double* linearD_n, 
double* linearD_s, 
double* linearD_r, 
double* linearD_t, 
double* linearMu_s, 
double* linearMu_r, 
double* linearMu_t, 
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
		contactForce[idx_c] = make_double3(0, 0, 0);
	    contactTorque[idx_c] = make_double3(0, 0, 0);

		const int idx_j = objectPointing[idx_c];
		const double rad_i = radius[idx_i];
		const size_t idx_w = wallIndex_t[idx_j];
		const double3 r_i = position[idx_i];
		const double3 r_w = position_w[idx_w];

		const double m_ij = 1. / inverseMass[idx_i];
		const double rad_ij = rad_i;

		const double3 p0 = globalVertices[vertIndex0_t[idx_j]];
		const double3 p1 = globalVertices[vertIndex1_t[idx_j]];
		const double3 p2 = globalVertices[vertIndex2_t[idx_j]];
		
		double3 r_c;
		SphereTriangleContactType type = classifySphereTriangleContact(r_i, rad_i,
									p0, p1, p2,
									r_c);
		if(type == SphereTriangleContactType::None) return;

		double3 n_ij = normalize(r_i - r_c);
		double delta = rad_i - length(r_i - r_c);

		if(type != SphereTriangleContactType::Face)
		{
			for(int idx_c1 = start; idx_c1 < idx_c; idx_c1++)
			{
				const int idx_j1 = objectPointing[idx_c1];
				const double3 p01 = globalVertices[vertIndex0_t[idx_j1]];
				const double3 p11 = globalVertices[vertIndex1_t[idx_j1]];
				const double3 p21 = globalVertices[vertIndex2_t[idx_j1]];

				double3 r_c1;
				SphereTriangleContactType type1 = classifySphereTriangleContact(r_i, rad_i,
									p01, p11, p21,
									r_c1);
				if(type1 == SphereTriangleContactType::None) continue;

				if(type == SphereTriangleContactType::Vertex)
				{
					if(length(r_c - r_c1) < 1.e-20) // share vertex, edge or face
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						return;
					}
				}
				else if(type == SphereTriangleContactType::Edge)
				{
					if(type1 == type)
					{
						if(length(r_c - r_c1) < 1.e-20) // share edge
						{
							slidingSpring[idx_c] = slidingSpring[idx_c1];
							rollingSpring[idx_c] = rollingSpring[idx_c1];
							torsionSpring[idx_c] = torsionSpring[idx_c1];
							return;
						}
					}
					else if(type1 == SphereTriangleContactType::Face) // share face
					{
						if(length(cross(r_c - p01, p11 - p01)) < 1.e-20) 
						{
							slidingSpring[idx_c] = slidingSpring[idx_c1];
							rollingSpring[idx_c] = rollingSpring[idx_c1];
							torsionSpring[idx_c] = torsionSpring[idx_c1];
							return;
						}
						if(length(cross(r_c - p11, p21 - p11)) < 1.e-20)
						{
							slidingSpring[idx_c] = slidingSpring[idx_c1];
							rollingSpring[idx_c] = rollingSpring[idx_c1];
							torsionSpring[idx_c] = torsionSpring[idx_c1];
							return;
						}
						if(length(cross(r_c - p01, p21 - p01)) < 1.e-20) 
						{
							slidingSpring[idx_c] = slidingSpring[idx_c1];
							rollingSpring[idx_c] = rollingSpring[idx_c1];
							torsionSpring[idx_c] = torsionSpring[idx_c1];
							return;
						}
					}
				}
			}
		}

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
		contactPoint[idx_c] = r_c;
		slidingSpring[idx_c] = epsilon_s;
		rollingSpring[idx_c] = epsilon_r;
		torsionSpring[idx_c] = epsilon_t;

		force[idx_i] += F_c;
		torque[idx_i] += T_c + cross(r_c - r_i, F_c);
	}
}

__global__ void triangleGlobalVerticesIntegrateKernel(double3* globalVertices, 
const double3* localVertices,
const int* triangleIndex_v,
const int* numTrianglesPrefixSum_v,
const int* wallIndex_t,
quaternion* orientation_w, 
double3* position_w,
const double dt,
const size_t numVertices)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numVertices) return;

	size_t idx_t = 0;
	if(idx_i > 0) idx_t = triangleIndex_v[numTrianglesPrefixSum_v[idx_i - 1]];
	size_t idx_w = wallIndex_t[idx_t];
	globalVertices[idx_i] = position_w[idx_w] + rotateVectorByQuaternion(orientation_w[idx_w], localVertices[idx_i]);
}

__global__ void orientationIntegrateKernel(quaternion* orientation, 
double3* angularVelocity, 
const double dt, 
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

	orientation[idx] = quaternionRotate(orientation[idx], angularVelocity[idx], dt);
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
    size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, balls.deviceSize(), maxThreadsPerBlock);
    calBallTriangleContactForceTorqueKernel <<<grid, block, 0, stream>>> (ballTriangleInteractions.force(),
	ballTriangleInteractions.torque(),
	ballTriangleInteractions.contactPoint(),
	ballTriangleInteractions.slidingSpring(),
	ballTriangleInteractions.rollingSpring(),
	ballTriangleInteractions.torsionSpring(),
	ballTriangleInteractions.objectPointing(),
	balls.force(),
	balls.torque(),
	balls.position(),
	balls.velocity(),
	balls.angularVelocity(),
	balls.radius(),
    balls.inverseMass(),
	balls.materialID(),
    ballTriangleInteractionMap.prefixSumA(),
    meshWalls.position(),
    meshWalls.velocity(),
    meshWalls.angularVelocity(),
    meshWalls.materialID(),
    meshWalls.triangles().wallIndex(),
    meshWalls.triangles().index0(),
    meshWalls.triangles().index1(),
    meshWalls.triangles().index2(),
    meshWalls.globalVertices(),
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

extern "C" void launchMeshWall1stHalfIntegration(meshWall &meshWalls, 
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
	size_t grid = 1, block = 1;

	computeGPUGridSizeBlockSize(grid, block, meshWalls.deviceSize(), maxThreadsPerBlock);
    positionIntegrationKernel <<<grid, block, 0, stream>>> (meshWalls.position(), 
    meshWalls.velocity(), 
    0.5 * timeStep, 
    meshWalls.deviceSize());

    orientationIntegrateKernel <<<grid, block, 0, stream>>> (meshWalls.orientation(), 
    meshWalls.angularVelocity(), 
    0.5 * timeStep, 
    meshWalls.deviceSize());

    computeGPUGridSizeBlockSize(grid, block, meshWalls.vertices().deviceSize(), maxThreadsPerBlock);
    triangleGlobalVerticesIntegrateKernel <<<grid, block, 0, stream>>> (meshWalls.globalVertices(),
    meshWalls.vertices().localPosition(),
    meshWalls.vertices().triangleIndex(),
    meshWalls.vertices().trianglesPrefixSum(),
    meshWalls.triangles().wallIndex(),
    meshWalls.orientation(),
    meshWalls.position(),
    0.5 * timeStep,
    meshWalls.vertices().deviceSize());
}

extern "C" void launchMeshWall2ndHalfIntegration(meshWall &meshWalls, 
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
	size_t grid = 1, block = 1;

    computeGPUGridSizeBlockSize(grid, block, meshWalls.vertices().deviceSize(), maxThreadsPerBlock);
    triangleGlobalVerticesIntegrateKernel <<<grid, block, 0, stream>>> (meshWalls.globalVertices(),
    meshWalls.vertices().localPosition(),
    meshWalls.vertices().triangleIndex(),
    meshWalls.vertices().trianglesPrefixSum(),
    meshWalls.triangles().wallIndex(),
    meshWalls.orientation(),
    meshWalls.position(),
    0.5 * timeStep,
    meshWalls.vertices().deviceSize());

    orientationIntegrateKernel <<<grid, block, 0, stream>>> (meshWalls.orientation(), 
    meshWalls.angularVelocity(), 
    0.5 * timeStep, 
    meshWalls.deviceSize());

	computeGPUGridSizeBlockSize(grid, block, meshWalls.deviceSize(), maxThreadsPerBlock);
    positionIntegrationKernel <<<grid, block, 0, stream>>> (meshWalls.position(), 
    meshWalls.velocity(), 
    0.5 * timeStep, 
    meshWalls.deviceSize());
}