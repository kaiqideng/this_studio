#include "DEMBallSolver.h"

bool DEMBallSolver::initialize() {
  std::cout << "DEM solver: initializing..." << std::endl;

  std::cout << "DEM solver: using GPU Device " << getGPUDeviceIndex() << std::endl;
  cudaError_t cudaStatus;
  cudaStatus = cudaSetDevice(getGPUDeviceIndex());
  if (cudaStatus != cudaSuccess) {
    std::cout << "DEM solver: cudaSetDevice( " << getGPUDeviceIndex()
              << " ) failed!  Do you have a CUDA-capable GPU installed?"
              << std::endl;
    exit(1);
  }
  std::cout << "DEM solver: downloading array from host to device..."
            << std::endl;
  ballInitialize(getDomainOrigin(), getDomainSize());
  ballNeighborSearch(getGPUThreadsPerBlock());
  handleHostArray();
  downloadContactModelParameters();
  outputBallVTU(dir_, iFrame_, iStep_, time_);
  std::cout << "DEM solver: initialization completed." << std::endl;
  return true;
}

void DEMBallSolver::solve() {
  removeVtuFiles(dir_);
  removeDatFiles(dir_);

  const double timeStep = getTimeStep();

  if (initialize()) 
  {
    size_t numSteps = size_t((getMaximumTime()) / timeStep) + 1;
    size_t frameInterval = numSteps / getNumFrames();
    if (frameInterval < 1)
      frameInterval = 1;

    while (iStep_ <= numSteps) {
      iStep_++;
      time_ += timeStep;
      ballNeighborSearch(getGPUThreadsPerBlock());
      ball1stHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
      ballContactCalculation(contactModelParams(), timeStep,getGPUThreadsPerBlock());
      if(handleHostArray()){
        ballInitialize(getDomainOrigin(), getDomainSize());
        downloadContactModelParameters();}
      ball2ndHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
      if (iStep_ % frameInterval == 0) {
        iFrame_++;
        std::cout << "DEM solver: frame " << iFrame_ << " at time " << time_
                  << std::endl;
        outputBallVTU(dir_, iFrame_, iStep_, time_);
      }
    }
  }
}