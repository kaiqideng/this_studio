#include "SPHIntegration.h"
#include "myStruct/interaction.h"

__global__ void calVelocityStarPositionStarKernel(double3* positionStar,
double3* velocityStar,
double3* position,
double3* velocity,
const double* mass,
const double* initialDensity,
const double* smoothLength,
const double* kinematicViscosity,
int* neighborPrifixSum_SPH,
int* objectPointing_SPH,
const double3 gravity,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 r_i = position[idx_i];
    double3 u_i = velocity[idx_i];
    double m_i = mass[idx_i];
    double rho0_i = initialDensity[idx_i];
    double h_i = smoothLength[idx_i];
    double nu_i = kinematicViscosity[idx_i];

    double3 viscosityTerm = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum_SPH[idx_i - 1];
    int end = neighborPrifixSum_SPH[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing_SPH[k];

        double3 r_j = position[idx_j];
        double3 u_j = velocity[idx_j];
        double m_j = mass[idx_j];
        double rho0_j = initialDensity[idx_j];
        double h_j = smoothLength[idx_j];
        double nu_j = kinematicViscosity[idx_j];

        double3 r_ij = r_i - r_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradWendlandKernel3D(r_ij, h_ij);

        double gamma = 0.001 * h_ij;
        viscosityTerm += m_j * 8.0 * (nu_i + nu_j) * dot(u_i - u_j, r_ij) * dW_ij 
        / ((rho0_i + rho0_j) * dot(r_ij, r_ij) + gamma * gamma);
    }

    velocityStar[idx_i] = velocity[idx_i] + (viscosityTerm + gravity) * timeStep;
    positionStar[idx_i] = position[idx_i] + velocityStar[idx_i] * timeStep;
}

__global__ void calDensityStarKernel(double* densityStar,
double3* positionStar,
double3* velocityStar,
const double* mass,
const double* initialDensity, 
const double* smoothLength,
int* neighborPrifixSum_SPH,
int* objectPointing_SPH,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 rs_i = positionStar[idx_i];
    double3 us_i = velocityStar[idx_i];
    double m_i = mass[idx_i];
    double h_i = smoothLength[idx_i];

    double dRho_i = 0.0;
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum_SPH[idx_i - 1];
    int end = neighborPrifixSum_SPH[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing_SPH[k];

        double3 rs_j = positionStar[idx_j];
        double3 us_j = velocityStar[idx_j];
        double m_j = mass[idx_j];
        double h_j = smoothLength[idx_j];

        double3 rs_ij = rs_i - rs_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradWendlandKernel3D(rs_ij, h_ij);

        dRho_i += m_j * dot(us_i - us_j, dW_ij);
    }

    densityStar[idx_i] = initialDensity[idx_i] + dRho_i * timeStep;
}

__global__ void calPresssureStarKernel(double* pressureStar,
double3* positionStar,
double3* velocityStar,
double* densityStar,
double* pressure,
const double* mass,
const double* initialDensity,
const double* smoothLength,
int* neighborPrifixSum_SPH,
int* objectPointing_SPH,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 rs_i = positionStar[idx_i];
    double3 us_i = velocityStar[idx_i];
    double rhos_i = densityStar[idx_i];
    double p_i = pressure[idx_i];
    double m_i = mass[idx_i];
    double h_i = smoothLength[idx_i];

    double B_i = 0.0;
    double A_i = 0.0;
    double AP_i = 0.0;
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum_SPH[idx_i - 1];
    int end = neighborPrifixSum_SPH[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing_SPH[k];

        double3 rs_j = positionStar[idx_j];
        double3 us_j = velocityStar[idx_j];
        double rhos_j = densityStar[idx_j];
        double p_j = pressure[idx_j];
        double m_j = mass[idx_j];
        double h_j = smoothLength[idx_j];

        double3 rs_ij = rs_i - rs_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradWendlandKernel3D(rs_ij, h_ij);

        double gamma = 0.001 * h_ij;
        double invRhos_ij = 1.0 / (rhos_i + rhos_j);
        double invRhos2_ij = invRhos_ij * invRhos_ij;
        double A_ij = m_j * 8.0 * invRhos2_ij * dot(rs_ij, dW_ij) 
        / (dot(rs_ij, rs_ij) + gamma * gamma);
        B_i += -m_j * dot(us_i - us_j, dW_ij) / rhos_j / timeStep;
        AP_i += A_ij * p_j;
        A_i += A_ij;
    }

    double ps_i = 0.0;
    if(fabs(A_i) > 1.e-20) ps_i = (B_i + AP_i) / A_i;
    if(ps_i < 0.0) ps_i = 0.0;
    pressureStar[idx_i] = ps_i;
}

__global__ void velocityPositionIntegrationKernel(double3* position,
double3* velocity,
double3* positionStar,
double3* velocityStar,
double* densityStar,
double* pressureStar,
const double* mass,
const double* initialDensity, 
const double* smoothLength,
int* neighborPrifixSum_SPH,
int* objectPointing_SPH,
double3* position_virtual,
double* effectiveRadius_virtual,
int* neighborPrifixSum_virtual,
int* objectPointing_virtual,
double3* force_SPHVirtual,
const double maximumAbsluteVelocity,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 r_i = position[idx_i];
    
    double3 rs_i = positionStar[idx_i];
    double rhos_i = densityStar[idx_i];
    double ps_i = pressureStar[idx_i];
    double m_i = mass[idx_i];
    double h_i = smoothLength[idx_i];

    double3 dp_rhos_i = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum_SPH[idx_i - 1];
    int end = neighborPrifixSum_SPH[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing_SPH[k];

        double3 rs_j = positionStar[idx_j];
        double rhos_j = densityStar[idx_j];
        double ps_j = pressureStar[idx_j];
        double m_j = mass[idx_j];
        double h_j = smoothLength[idx_j];

        double3 rs_ij = rs_i - rs_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradWendlandKernel3D(rs_ij, h_ij);

        dp_rhos_i += -m_j * (ps_i / (rhos_i * rhos_i) + ps_j / (rhos_j * rhos_j)) * dW_ij;
    }

    double3 PB_i = make_double3(0.0, 0.0, 0.0);
    start = 0;
    if (idx_i > 0) start = neighborPrifixSum_virtual[idx_i - 1];
    end = neighborPrifixSum_virtual[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_v = objectPointing_virtual[k];

        double3 r_v = position_virtual[idx_v];
        double D = maximumAbsluteVelocity * maximumAbsluteVelocity;
        double r2 = effectiveRadius_virtual[idx_v] * effectiveRadius_virtual[idx_v];
        double3 r_iv = r_i - r_v;
        double r2_iv = dot(r_iv, r_iv);
        double3 PB_iv = make_double3(0.0, 0.0, 0.0);
        if(r2_iv > 1.e-20 && r2 / r2_iv >= 1.0)
        {
            PB_iv = D * (pow(r2 / r2_iv, 6.0) - pow(r2 / r2_iv, 2.0)) / r2_iv * r_iv;
            PB_i += PB_iv;
        }
        force_SPHVirtual[k] = PB_iv * m_i;
    }

    double3 u_i = velocity[idx_i];
    double3 u1_i = velocityStar[idx_i] + dp_rhos_i * timeStep;
    u1_i += PB_i * timeStep;
    velocity[idx_i] = u1_i;
    position[idx_i] += 0.5 * (u_i + u1_i) * timeStep;
}

extern "C" void launchSPHIntegration(SPH& SPHs, 
SPHInteraction& SPHInteractions, 
interactionMap &SPHInteractionMap,
virtualParticle& virtualParticles, 
SPHInteraction& SPHVirtualInteractions, 
interactionMap &SPHVirtualInteractionMap,
const double maximumAbsluteVelocity,
const double3 gravity,
const double timeStep,
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, SPHs.deviceSize(), maxThreadsPerBlock);

    calVelocityStarPositionStarKernel <<<grid, block, 0, stream>>> (SPHs.positionStar(),
    SPHs.velocityStar(),
    SPHs.position(),
    SPHs.velocity(),
    SPHs.mass(),
    SPHs.initialDensity(),
    SPHs.smoothLength(),
    SPHs.kinematicViscosity(), 
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
   
    gravity,
    timeStep, 
    SPHs.deviceSize());

    calDensityStarKernel <<<grid, block, 0, stream>>> (SPHs.densityStar(),
    SPHs.positionStar(),
    SPHs.velocityStar(),
    SPHs.mass(),
    SPHs.initialDensity(),
    SPHs.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    timeStep, 
    SPHs.deviceSize());

    calPresssureStarKernel <<<grid, block, 0, stream>>> (SPHs.pressureStar(),
    SPHs.positionStar(),
    SPHs.velocityStar(),
    SPHs.densityStar(),
    SPHs.pressure(),
    SPHs.mass(),
    SPHs.initialDensity(),
    SPHs.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    timeStep, 
    SPHs.deviceSize());

    cuda_copy(SPHs.pressure(), SPHs.pressureStar(), SPHs.deviceSize(), CopyDir::D2D, stream);

    velocityPositionIntegrationKernel <<<grid, block, 0, stream>>> (SPHs.position(),
    SPHs.velocity(),
    SPHs.positionStar(),
    SPHs.velocityStar(),
    SPHs.densityStar(),
    SPHs.pressureStar(),
    SPHs.mass(),
    SPHs.initialDensity(),
    SPHs.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    virtualParticles.position(),
    virtualParticles.effectiveRadius(),
    SPHVirtualInteractionMap.prefixSumA(),
    SPHVirtualInteractions.objectPointing(),
    SPHVirtualInteractions.force(),
    maximumAbsluteVelocity,
    timeStep, 
    SPHs.deviceSize());
}