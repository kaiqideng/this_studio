#include "SPHIntegration.h"

__global__ void calDummyParticleNormalKernel(double3* normal,
double3* position,
const double* density,
const double* mass,
const double* smoothLength,
int* neighborPrifixSum,
int* objectPointing,
const size_t numSPHs,
const size_t numDummy)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numDummy) return;
    idx_i += numSPHs;
    normal[idx_i] = make_double3(0., 0., 0.);

    double3 r_i = position[idx_i];
    double h_i = smoothLength[idx_i];

    double3 Theta = make_double3(0., 0., 0.);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        if (idx_j >= numSPHs)
        {
            double3 r_j = position[idx_j];
            double h_j = smoothLength[idx_j];
            double m_j = mass[idx_j];
            double rho_j = density[idx_j];

            double3 r_ij = r_i - r_j;
            double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

            if (rho_j > 0.0) Theta -= m_j / rho_j * dW_ij;
        }
    }

    if (lengthSquared(Theta) > 1.e-10) normal[idx_i] = Theta / length(Theta);
}

extern "C" void launchCalDummyParticleNormal(WCSPH& WCSPHs, 
SPHInteraction& SPHInteractions,
interactionMap& SPHInteractionMap,
const size_t maxThreadPerBlock,
cudaStream_t stream)
{
    size_t gridDim = 1, blockDim = 1;
    if (setGPUGridBlockDim(gridDim, blockDim, WCSPHs.dummyDeviceSize(), maxThreadPerBlock))
    {
        calDummyParticleNormalKernel <<<gridDim, blockDim, 0, stream>>> (WCSPHs.normal(), 
        WCSPHs.position(), 
        WCSPHs.density(), 
        WCSPHs.mass(), 
        WCSPHs.smoothLength(), 
        SPHInteractionMap.prefixSumA(), 
        SPHInteractions.objectPointing(),
        WCSPHs.SPHDeviceSize(), 
        WCSPHs.dummyDeviceSize());
    }
}

__global__ void updateWCSPHDensityKernel(double* density,
double* pressure,
double3* position,
double3* velocity,
const double* soundSpeed,
const double* mass,
const double* initialDensity,
const double* smoothLength,
double3* normal,
int* neighborPrifixSum,
int* objectPointing,
const double3 gravity,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 r_i = position[idx_i];
    double3 v_i = velocity[idx_i];
    double c_i = soundSpeed[idx_i];
    double h_i = smoothLength[idx_i];
    double rho_i = density[idx_i];
    double P_L = pressure[idx_i];

    double dRho = 0.0;
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double3 v_j = velocity[idx_j];
        double c_j = soundSpeed[idx_j];
        double h_j = smoothLength[idx_j];
        double rho_j = density[idx_j];
        double P_R = pressure[idx_j];
        double m_j = mass[idx_j];

        double3 r_ij = r_i - r_j;
        double3 e_ij = -normalize(r_ij);
        double U_L = dot(v_i, e_ij);
        double U_R = dot(v_j, e_ij);

        if (idx_j >= numSPHs) // wall
        {
            double3 n_w = normal[idx_j];
            U_L = dot(-n_w, v_i);
            U_R = -U_L + 2.0 * dot(-n_w, v_j);
            P_R = P_L + rho_i * dot(gravity, r_j - r_i);
            //e_ij = -n_w;
            //v_j += (U_R - dot(v_j, e_ij)) * e_ij;
            if (c_j > 0.0) rho_j = P_R / (c_j * c_j) + initialDensity[idx_j];
        }

        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double3 v_star = 0.5 * (v_i + v_j);
        if (rho_bar > 0.0 && c_bar > 0.0) v_star += 0.5 * ((P_L - P_R) / (rho_bar * c_bar)) * e_ij;

        if (rho_j > 0.0) dRho += m_j / rho_j * dot(v_i - v_star, dW_ij);
    }

    dRho *= 2.0 * rho_i;
    density[idx_i] += dRho * timeStep;
}

__global__ void calWCSPHPressureKernel(double* pressure,
double* density,
const double* soundSpeed,
const double* initialDensity,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double c = soundSpeed[idx_i];
    pressure[idx_i] = c * c * (density[idx_i] - initialDensity[idx_i]);
}

__global__ void updateWCSPHVelocityKernel(double3* velocity,
double* density,
double* pressure,
double3* position,
const double* soundSpeed,
const double* mass,
const double* initialDensity,
const double* smoothLength,
double3* normal,
int* neighborPrifixSum,
int* objectPointing,
const double3 gravity,
const double timeStep,
const size_t numSPHs)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPHs) return;

    double3 r_i = position[idx_i];
    double3 v_i = velocity[idx_i];
    double c_i = soundSpeed[idx_i];
    double h_i = smoothLength[idx_i];
    double rho_i = density[idx_i];
    double P_L = pressure[idx_i];

    double3 acc = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double3 v_j = velocity[idx_j];
        double c_j = soundSpeed[idx_j];
        double h_j = smoothLength[idx_j];
        double rho_j = density[idx_j];
        double P_R = pressure[idx_j];

        double m_j = mass[idx_j];

        double3 r_ij = r_i - r_j;
        double3 e_ij = -normalize(r_ij);
        double U_L = dot(v_i, e_ij);
        double U_R = dot(v_j, e_ij);

        if (idx_j >= numSPHs)
        {
            double3 n_w = normal[idx_j];
            U_L = dot(-n_w, v_i);
            U_R = -U_L + 2 * dot(-n_w, v_j);
            P_R = P_L + rho_i * dot(gravity, r_j - r_i);
            //e_ij = -n_w;
            //v_j += (U_R - dot(v_j, e_ij)) * e_ij;
            if (c_j > 0.0) rho_j = P_R / (c_j * c_j) + initialDensity[idx_j];
        }

        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double beta = fmin(3.0 * fmax(U_L - U_R, 0.0), c_bar);
        double P_star = 0.5 * (P_L + P_R) + 0.5 * beta * rho_bar * (U_L - U_R);
        if (rho_i > 0.0 && rho_j > 0.0) acc -= 2.0 * m_j * (P_star / (rho_i * rho_j)) * dW_ij;
    }

    velocity[idx_i] += (acc + gravity) * timeStep;
}

extern "C" void launchWCSPH1stIntegration(WCSPH& WCSPHs, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
    updateWCSPHVelocityKernel <<<gridDim, blockDim, 0, stream>>> (WCSPHs.velocity(),
    WCSPHs.density(),
    WCSPHs.pressure(),
    WCSPHs.position(),
    WCSPHs.soundSpeed(),
    WCSPHs.mass(),
    WCSPHs.initialDensity(),
    WCSPHs.smoothLength(),
    WCSPHs.normal(), 
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    gravity,
    0.5 * timeStep, 
    WCSPHs.SPHDeviceSize());

    positionIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (WCSPHs.position(), 
    WCSPHs.velocity(), 
    timeStep, 
    WCSPHs.SPHDeviceSize());
}

extern "C" void launchWCSPH2ndIntegration(WCSPH& WCSPHs, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
    updateWCSPHDensityKernel <<<gridDim, blockDim, 0, stream>>> (WCSPHs.density(),
    WCSPHs.pressure(),
    WCSPHs.position(),
    WCSPHs.velocity(),
    WCSPHs.soundSpeed(),
    WCSPHs.mass(),
    WCSPHs.initialDensity(),
    WCSPHs.smoothLength(),
    WCSPHs.normal(), 
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    gravity,
    timeStep, 
    WCSPHs.SPHDeviceSize());

    calWCSPHPressureKernel <<<gridDim, blockDim, 0, stream>>> (WCSPHs.pressure(),
    WCSPHs.density(),
    WCSPHs.soundSpeed(),
    WCSPHs.initialDensity(),
    WCSPHs.SPHDeviceSize());

    updateWCSPHVelocityKernel <<<gridDim, blockDim, 0, stream>>> (WCSPHs.velocity(),
    WCSPHs.density(),
    WCSPHs.pressure(),
    WCSPHs.position(),
    WCSPHs.soundSpeed(),
    WCSPHs.mass(),
    WCSPHs.initialDensity(),
    WCSPHs.smoothLength(),
    WCSPHs.normal(), 
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    gravity,
    0.5 * timeStep, 
    WCSPHs.SPHDeviceSize());
}

//EISPH
__global__ void calISPHVelocityStarPositionStarKernel(double3* positionStar,
double3* velocityStar,
double3* position,
double3* velocity,
const double* mass,
const double* initialDensity,
const double* smoothLength,
const double* kinematicViscosity,
int* neighborPrifixSum,
int* objectPointing,
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

__global__ void calISPHDensityStarKernel(double* densityStar,
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

__global__ void calISPHPresssureStarKernel(double* pressureStar,
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

__global__ void ISPHVelocityPositionIntegrationKernel(double3* position,
double3* velocity,
double3* positionStar,
double3* velocityStar,
double* densityStar,
double* pressureStar,
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
    double ps_i = pressureStar[idx_i];

    double3 dp_rhos_i = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double rhos_j = densityStar[idx_j];
        double ps_j = pressureStar[idx_j];
        double m_j = mass[idx_j];
        double3 dW_ij = gradientKernelStar[k];

        dp_rhos_i += -m_j * (ps_i / (rhos_i * rhos_i) + ps_j / (rhos_j * rhos_j)) * dW_ij;
    }

    double3 u_i = velocity[idx_i];
    double3 u1_i = velocityStar[idx_i] + dp_rhos_i * timeStep;
    velocity[idx_i] = u1_i;
    position[idx_i] += 0.5 * (u_i + u1_i) * timeStep;
}

__global__ void setAdamiDummyPressureKernel(double* pressure,
double3* acceleration, 
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
        WRho_i += W_ij * rho_j * r_ij;
    }

    if (W_i > 1.e-20)
    {
        pressure[idx_i] = (WP_i + dot(gravity - acceleration[idx_i], WRho_i)) / W_i;
        if (pressure[idx_i] < 0.0) pressure[idx_i] = 0.0;
    }
}

extern "C" void launchISPH1stIntegration(ISPH& ISPHs, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double3 gravity,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
    calISPHVelocityStarPositionStarKernel <<<gridDim, blockDim, 0, stream>>> (ISPHs.positionStar(),
    ISPHs.velocityStar(),
    ISPHs.position(),
    ISPHs.velocity(),
    ISPHs.mass(),
    ISPHs.initialDensity(),
    ISPHs.smoothLength(),
    ISPHs.kinematicViscosity(), 
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    gravity,
    timeStep, 
    ISPHs.SPHDeviceSize());
}

extern "C" void launchISPH2ndIntegration(ISPH& ISPHs, 
SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
const double timeStep,
const size_t gridDim,
const size_t blockDim, 
cudaStream_t stream)
{
    size_t numSPHs = ISPHs.SPHDeviceSize();

    calISPHDensityStarKernel <<<gridDim, blockDim, 0, stream>>> (ISPHs.densityStar(),
    ISPHs.initialDensity(),
    ISPHs.positionStar(),
    ISPHs.velocityStar(),
    ISPHs.mass(),
    ISPHs.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernelStar(),
    timeStep, 
    numSPHs);

    calISPHPresssureStarKernel <<<gridDim, blockDim, 0, stream>>> (ISPHs.pressureStar(),
    ISPHs.positionStar(),
    ISPHs.velocityStar(),
    ISPHs.densityStar(),
    ISPHs.pressure(),
    ISPHs.mass(),
    ISPHs.smoothLength(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernelStar(),
    timeStep, 
    numSPHs);

    cuda_copy(ISPHs.pressure(), ISPHs.pressureStar(), numSPHs, CopyDir::D2D, stream);

    ISPHVelocityPositionIntegrationKernel <<<gridDim, blockDim, 0, stream>>> (ISPHs.position(),
    ISPHs.velocity(),
    ISPHs.positionStar(),
    ISPHs.velocityStar(),
    ISPHs.densityStar(),
    ISPHs.pressureStar(),
    ISPHs.mass(),
    SPHInteractionMap.prefixSumA(),
    SPHInteractions.objectPointing(),
    SPHInteractions.gradientKernelStar(),
    timeStep, 
    ISPHs.SPHDeviceSize());
}