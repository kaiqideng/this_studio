#include "WCSPHIntegrationKernel.h"
#include "myUtility/myVec.h"

__global__ void calDummyParticleNormalKernel(double3* normal,
const double3* position,
const double* density,
const double* mass,
const double* smoothLength,
const int* neighborPrifixSum,
const int* objectPointing,
const size_t numDummy)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numDummy) return;
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

        double3 r_j = position[idx_j];
        double h_j = smoothLength[idx_j];
        double m_j = mass[idx_j];
        double rho_j = density[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        if (rho_j > 0.0) Theta -= m_j / rho_j * dW_ij;
    }

    if (lengthSquared(Theta) > 1.e-10) normal[idx_i] = Theta / length(Theta);
}

__global__ void updateWCSPHDensityKernel(double* density,
const double* pressure,
const double3* position,
const double3* velocity,
const double* soundSpeed,
const double* mass,
const double* initialDensity,
const double* smoothLength,
const int* neighborPrifixSum,
const double3* position_dummy,
const double3* velocity_dummy,
const double3* normal_dummy,
const double* soundSpeed_dummy,
const double* mass_dummy,
const double* initialDensity_dummy,
const double* smoothLength_dummy,
const int* neighborPrifixSum_dummy,
const int* objectPointing,
const int* objectPointing_dummy,
const double3 gravity,
const double timeStep,
const size_t numSPH)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

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
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));
        double3 e_ij = -normalize(r_ij);

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double3 v_star = 0.5 * (v_i + v_j);
        if (rho_bar > 0.0 && c_bar > 0.0) v_star += 0.5 * ((P_L - P_R) / (rho_bar * c_bar)) * e_ij;
        if (rho_j > 0.0) dRho += m_j / rho_j * dot(v_i - v_star, dW_ij);
    }

    if (idx_i > 0) start = neighborPrifixSum_dummy[idx_i - 1];
    end = neighborPrifixSum_dummy[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing_dummy[k];

        double3 r_j = position_dummy[idx_j];
        double3 v_j = velocity_dummy[idx_j];
        double c_j = soundSpeed_dummy[idx_j];
        double h_j = smoothLength_dummy[idx_j];
        double rho_j = initialDensity_dummy[idx_j];
        double m_j = mass_dummy[idx_j];

        double P_R = P_L + rho_i * dot(gravity, r_j - r_i);
        if (c_j > 0.0) rho_j = P_R / (c_j * c_j) + initialDensity_dummy[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));
        double3 e_ij = -normal_dummy[idx_j];

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double3 v_star = 0.5 * (v_i + v_j);
        if (rho_bar > 0.0 && c_bar > 0.0) v_star += 0.5 * ((P_L - P_R) / (rho_bar * c_bar)) * e_ij;
        if (rho_j > 0.0) dRho += m_j / rho_j * dot(v_i - v_star, dW_ij);
    }

    dRho *= 2.0 * rho_i;
    density[idx_i] += dRho * timeStep;
}

__global__ void SPHPositionIntegrationKernel(double3* position,
const double3* velocity,
const double timeStep,
const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	position[idx_i] += timeStep * velocity[idx_i];
}

__global__ void calWCSPHPressureKernel(double* pressure,
const double* density,
const double* soundSpeed,
const double* initialDensity,
const size_t numSPH)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

    double c = soundSpeed[idx_i];
    pressure[idx_i] = c * c * (density[idx_i] - initialDensity[idx_i]);
}

__global__ void updateWCSPHVelocityKernel(double3* velocity,
const double3* position,
const double* density,
const double* pressure,
const double* soundSpeed,
const double* mass,
const double* initialDensity,
const double* smoothLength,
const int* neighborPrifixSum,
const double3* position_dummy,
const double3* velocity_dummy,
const double3* normal_dummy,
const double* soundSpeed_dummy,
const double* mass_dummy,
const double* initialDensity_dummy,
const double* smoothLength_dummy,
const int* neighborPrifixSum_dummy,
const int* objectPointing,
const int* objectPointing_dummy,
const double3 gravity,
const double timeStep,
const size_t numSPH)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

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
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double beta = fmin(3.0 * fmax(U_L - U_R, 0.0), c_bar);
        double P_star = 0.5 * (P_L + P_R) + 0.5 * beta * rho_bar * (U_L - U_R);
        if (rho_i > 0.0 && rho_j > 0.0) acc -= 2.0 * m_j * (P_star / (rho_i * rho_j)) * dW_ij;
    }

    if (idx_i > 0) start = neighborPrifixSum_dummy[idx_i - 1];
    end = neighborPrifixSum_dummy[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing_dummy[k];

        double3 r_j = position_dummy[idx_j];
        double3 v_j = velocity_dummy[idx_j];
        double c_j = soundSpeed_dummy[idx_j];
        double h_j = smoothLength_dummy[idx_j];
        double rho_j = initialDensity_dummy[idx_j];
        double m_j = mass_dummy[idx_j];

        double3 n_w = normal_dummy[idx_j];
        double U_L = dot(-n_w, v_i);
        double U_R = -U_L + 2.0 * dot(-n_w, v_j);
        double P_R = P_L + rho_i * dot(gravity, r_j - r_i);
        if (c_j > 0.0) rho_j = P_R / (c_j * c_j) + initialDensity_dummy[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double beta = fmin(3.0 * fmax(U_L - U_R, 0.0), c_bar);
        double P_star = 0.5 * (P_L + P_R) + 0.5 * beta * rho_bar * (U_L - U_R);
        if (rho_i > 0.0 && rho_j > 0.0) acc -= 2.0 * m_j * (P_star / (rho_i * rho_j)) * dW_ij;
    }

    velocity[idx_i] += (acc + gravity) * timeStep;
}

extern "C" void launchCalDummyParticleNormal(double3* normal,
double3* position,
double* density,
double* mass,
double* smoothLength,
int* neighborPrifixSum,

int* objectPointing,

const size_t numDummy,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    if (gridD_GPU * blockD_GPU < numDummy) return;

    calDummyParticleNormalKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (normal, 
    position, 
    density, 
    mass, 
    smoothLength, 
    neighborPrifixSum, 
    objectPointing,
    numDummy);
}

extern "C" void launchWCSPH1stHalfIntegration(double3* position,
double3* velocity,
double* density,
double* pressure,
double* soundSpeed,
double* mass,
double* initialDensity,
double* smoothLength,
int* neighborPrifixSum,

double3* position_dummy,
double3* velocity_dummy,
double3* normal_dummy,
double* soundSpeed_dummy,
double* mass_dummy,
double* initialDensity_dummy,
double* smoothLength_dummy,
int* neighborPrifixSum_dummy,

int* objectPointing,

int* objectPointing_dummy,

const double3 gravity,
const double timeStep,

const size_t numSPH,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    if (gridD_GPU * blockD_GPU < numSPH) return;

    updateWCSPHVelocityKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (velocity,
    position,
    density,
    pressure,
    soundSpeed,
    mass,
    initialDensity,
    smoothLength,
    neighborPrifixSum,
    position_dummy,
    velocity_dummy,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    neighborPrifixSum_dummy,
    objectPointing,
    objectPointing_dummy,
    gravity,
    0.5 * timeStep, 
    numSPH);

    SPHPositionIntegrationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (position, 
    velocity, 
    timeStep, 
    numSPH);
}

extern "C" void launchWCSPH2ndHalfIntegration(double3* position,
double3* velocity,
double* density,
double* pressure,
double* soundSpeed,
double* mass,
double* initialDensity,
double* smoothLength,
int* neighborPrifixSum,

double3* position_dummy,
double3* velocity_dummy,
double3* normal_dummy,
double* soundSpeed_dummy,
double* mass_dummy,
double* initialDensity_dummy,
double* smoothLength_dummy,
int* neighborPrifixSum_dummy,

int* objectPointing,

int* objectPointing_dummy,

const double3 gravity,
const double timeStep,

const size_t numSPH,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    if (gridD_GPU * blockD_GPU < numSPH) return;

    updateWCSPHDensityKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (density,
    pressure,
    position,
    velocity,
    soundSpeed,
    mass,
    initialDensity,
    smoothLength,
    neighborPrifixSum,
    position_dummy,
    velocity_dummy,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    neighborPrifixSum_dummy,
    objectPointing,
    objectPointing_dummy,
    gravity,
    timeStep, 
    numSPH);

    calWCSPHPressureKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (pressure,
    density,
    soundSpeed,
    initialDensity,
    numSPH);

    updateWCSPHVelocityKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (velocity,
    position,
    density,
    pressure,
    soundSpeed,
    mass,
    initialDensity,
    smoothLength,
    neighborPrifixSum,
    position_dummy,
    velocity_dummy,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    neighborPrifixSum_dummy,
    objectPointing,
    objectPointing_dummy,
    gravity,
    0.5 * timeStep, 
    numSPH);
}