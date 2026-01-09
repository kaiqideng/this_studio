#include "SPHIntegration.h"

__global__ void calVelocityStarPositionStarKernel(double3* positionStar,
double3* velocityStar,
double3* position,
double3* velocity,
const double* mass,
const double* initialDensity,
const double* smoothLength,
const double* kinematicViscosity,
int* neighborPrifixSum,
int* objectPointing,
double3* gradientKernel,
const double3 gravity,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 r_i = position[idx_i];
    double3 u_i = velocity[idx_i];
    double rho0_i = initialDensity[idx_i];
    double h_i = smoothLength[idx_i];
    double nu_i = kinematicViscosity[idx_i];

    double3 viscosityTerm = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double3 u_j = velocity[idx_j];
        double m_j = mass[idx_j];
        double rho0_j = initialDensity[idx_j];
        double h_j = smoothLength[idx_j];
        double nu_j = kinematicViscosity[idx_j];
        if(idx_j >= numSPHs) nu_j = nu_i;

        double3 r_ij = r_i - r_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradWendlandKernel3D(r_ij, h_ij);
        
        gradientKernel[k] = dW_ij;

        double gamma = 0.001 * h_ij;
        viscosityTerm += m_j * 8.0 * (nu_i + nu_j) * dot(u_i - u_j, r_ij) * dW_ij 
        / ((rho0_i + rho0_j) * dot(r_ij, r_ij) + gamma * gamma);
    }

    velocityStar[idx_i] = velocity[idx_i] + (viscosityTerm + gravity) * timeStep;
    positionStar[idx_i] = position[idx_i] + velocityStar[idx_i] * timeStep;
}

__global__ void calDensityStarKernel(double* densityStar,
const double* initialDensity, 
double3* positionStar,
double3* velocityStar,
const double* mass,
const double* smoothLength,
int* neighborPrifixSum,
int* objectPointing,
double3* gradientKernelStar,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 rs_i = positionStar[idx_i];
    double3 us_i = velocityStar[idx_i];
    double h_i = smoothLength[idx_i];

    double dRho_i = 0.0;
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 rs_j = positionStar[idx_j];
        double3 us_j = velocityStar[idx_j];
        double m_j = mass[idx_j];
        double h_j = smoothLength[idx_j];

        double3 rs_ij = rs_i - rs_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradWendlandKernel3D(rs_ij, h_ij);

        gradientKernelStar[k] = dW_ij;

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
const double* smoothLength,
int* neighborPrifixSum,
int* objectPointing,
double3* gradientKernelStar,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 rs_i = positionStar[idx_i];
    double3 us_i = velocityStar[idx_i];
    double rhos_i = densityStar[idx_i];
    double h_i = smoothLength[idx_i];

    double B_i = 0.0;
    double A_i = 0.0;
    double AP_i = 0.0;
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 rs_j = positionStar[idx_j];
        double3 us_j = velocityStar[idx_j];
        double rhos_j = densityStar[idx_j];
        double p_j = pressure[idx_j];
        double m_j = mass[idx_j];
        double h_j = smoothLength[idx_j];

        double3 rs_ij = rs_i - rs_j;
        double h_ij = 0.5 * (h_i + h_j);
        double3 dW_ij = gradientKernelStar[k];

        double gamma = 0.001 * h_ij;
        double invRhos_ij = 1.0 / (rhos_i + rhos_j);
        double invRhos2_ij = invRhos_ij * invRhos_ij;
        double A_ij = m_j * 8.0 * invRhos2_ij * dot(rs_ij, dW_ij) 
        / (dot(rs_ij, rs_ij) + gamma * gamma);
        B_i += -m_j * dot(us_i - us_j, dW_ij) / rhos_j;
        AP_i += A_ij * p_j;
        A_i += A_ij;
    }

    B_i /= timeStep;
    double ps_i = 0.0;
    if (fabs(A_i) > 1.e-20) ps_i = (B_i + AP_i) / A_i;
    if (ps_i < 0.0) ps_i = 0.0;
    pressureStar[idx_i] = ps_i;
}

__global__ void velocityPositionIntegrationKernel(double3* position,
double3* velocity,
double3* positionStar,
double3* velocityStar,
double* densityStar,
double* pressure,
const double* mass,
int* neighborPrifixSum,
int* objectPointing,
double3* gradientKernelStar,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double rhos_i = densityStar[idx_i];
    double p_i = pressure[idx_i];

    double3 dp_rhos_i = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double rhos_j = densityStar[idx_j];
        double p_j = pressure[idx_j];
        double m_j = mass[idx_j];
        double3 dW_ij = gradientKernelStar[k];

        dp_rhos_i += -m_j * (p_i / (rhos_i * rhos_i) + p_j / (rhos_j * rhos_j)) * dW_ij;
    }

    double3 u_i = velocity[idx_i];
    double3 u1_i = velocityStar[idx_i] + dp_rhos_i * timeStep;
    velocity[idx_i] = u1_i;
    position[idx_i] += 0.5 * (u_i + u1_i) * timeStep;
}

__global__ void setDummyPressureKernel(double* pressure, 
double3* velocity, 
double3* position, 
const double* initialDensity,
const double* smoothLength,
int* neighborPrifixSum,
int* objectPointing,
const double3 gravity,
const size_t numSPHs,
const size_t numGhosts)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numGhosts) return;
    idx_i += numSPHs;

    pressure[idx_i] = 0.0;

    double3 r_i = position[idx_i];
    double h_i = smoothLength[idx_i];

    double W_i = 0.0;
    double WP_i = 0.0;
    double3 WRho_i = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double h_j = smoothLength[idx_j];
        double3 r_ij = r_i - r_j;
        double h_ij = 0.5 * (h_i + h_j);
        double W_ij = wendlandKernel3D(length(r_ij), h_ij);
        W_i += W_ij;
        double p_j = pressure[idx_j];
        double rho_j = initialDensity[idx_j];
        WP_i += W_ij * p_j;
        WRho_i = W_ij * rho_j * r_ij;
    }

    if(W_i > 1.e-20)
    {
        pressure[idx_i] = (WP_i + dot(gravity, WRho_i)) / W_i;
    }
}

extern "C" void launchSPH1stIntegration(SPH& SPHAndGhosts, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
    calVelocityStarPositionStarKernel <<<gridDim, blockDim, 0, stream>>> (SPHAndGhosts.positionStar(),
    SPHAndGhosts.velocityStar(),
    SPHAndGhosts.position(),
    SPHAndGhosts.velocity(),
    SPHAndGhosts.mass(),
    SPHAndGhosts.initialDensity(),
    SPHAndGhosts.smoothLength(),
    SPHAndGhosts.kinematicViscosity(), 
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernel(),
    gravity,
    timeStep, 
    SPHAndGhosts.SPHDeviceSize());
}

extern "C" void launchSPH2ndIntegration(SPH& SPHAndGhosts, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
    size_t numSPHs = SPHAndGhosts.SPHDeviceSize();

    calDensityStarKernel <<<gridDim, blockDim, 0, stream>>> (SPHAndGhosts.densityStar(),
    SPHAndGhosts.initialDensity(),
    SPHAndGhosts.positionStar(),
    SPHAndGhosts.velocityStar(),
    SPHAndGhosts.mass(),
    SPHAndGhosts.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernelStar(),
    timeStep, 
    numSPHs);

    calPresssureStarKernel <<<gridDim, blockDim, 0, stream>>> (SPHAndGhosts.pressureStar(),
    SPHAndGhosts.positionStar(),
    SPHAndGhosts.velocityStar(),
    SPHAndGhosts.densityStar(),
    SPHAndGhosts.pressure(),
    SPHAndGhosts.mass(),
    SPHAndGhosts.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernelStar(),
    timeStep, 
    numSPHs);

    cuda_copy(SPHAndGhosts.pressure(), SPHAndGhosts.pressureStar(), numSPHs, CopyDir::D2D, stream);
}

extern "C" void launchSPH3rdIntegration(SPH& SPHAndGhosts, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double timeStep,
const size_t gridDim,
const size_t blockDim,
cudaStream_t stream)
{
    velocityPositionIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (SPHAndGhosts.position(),
    SPHAndGhosts.velocity(),
    SPHAndGhosts.positionStar(),
    SPHAndGhosts.velocityStar(),
    SPHAndGhosts.densityStar(),
    SPHAndGhosts.pressure(),
    SPHAndGhosts.mass(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernelStar(),
    timeStep, 
    SPHAndGhosts.SPHDeviceSize());
}

extern "C" void launchAdamiBoundaryCondition(SPH& SPHAndGhosts, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim,
cudaStream_t stream)
{
    size_t numSPHs = SPHAndGhosts.SPHDeviceSize();
    size_t numGhosts = SPHAndGhosts.ghostDeviceSize();
    cuda_copy(SPHAndGhosts.positionStar() + numSPHs, SPHAndGhosts.position() + numSPHs, numGhosts, CopyDir::D2D, stream);
    cuda_copy(SPHAndGhosts.velocityStar() + numSPHs, SPHAndGhosts.velocity() + numSPHs, numGhosts, CopyDir::D2D, stream);
    cuda_copy(SPHAndGhosts.densityStar() + numSPHs, SPHAndGhosts.initialDensity() + numSPHs, numGhosts, CopyDir::D2D, stream);

    setDummyPressureKernel <<<gridDim, blockDim, 0, stream>>> (SPHAndGhosts.pressure(), 
    SPHAndGhosts.velocity(), 
    SPHAndGhosts.position(), 
    SPHAndGhosts.initialDensity(), 
    SPHAndGhosts.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    gravity,
    numSPHs,
    numGhosts);
}