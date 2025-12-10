#include "DEMSolver.h"
#include "myContainer/myUtility/myFileEdit.h"

bool DEMSolver::initialize() {
  std::cout << "DEM solver: Initializing..." << std::endl;

  std::cout << "DEM solver: Using GPU Device " << getGPUDeviceIndex() << std::endl;
  cudaError_t cudaStatus;
  cudaStatus = cudaSetDevice(getGPUDeviceIndex());
  if (cudaStatus != cudaSuccess) {
    std::cout << "DEM solver: Function cudaSetDevice( " << getGPUDeviceIndex()
              << " ) failed!  Do you have a CUDA-capable GPU installed?"
              << std::endl;
    exit(1);
  }
  std::cout << "DEM solver: Downloading array from host to device..."
            << std::endl;
  DEMInitialize(getDomainOrigin(), getDomainSize());

  std::cout << "DEM solver: Initialization completed." << std::endl;
  return true;
}

void DEMSolver::solve() {
  removeVtuFiles(getDir());
  removeDatFiles(getDir());

  const double timeStep = getTimeStep();

  if (initialize()) {
    numSteps = size_t((getMaximumTime()) / timeStep) + 1;
    frameInterval = numSteps / getNumFrames();
    if (frameInterval < 1)
      frameInterval = 1;
    while (iStep <= numSteps) {
      iStep++;
      DEMUpdate(getDomainOrigin(), getDomainSize(), getGravity(),
                timeStep, getGPUThreadsPerBlock());
      if (iStep % frameInterval == 0) {
        iFrame++;
        std::cout << "DEM solver: Frame " << iFrame << " at time " << iStep * timeStep
                  << std::endl;
        outputSolidParticleVTU(iFrame, iStep, timeStep);
      }
    }
  }
}