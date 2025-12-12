#include "DEMSolver.h"
#include "myContainer/myWall.h"

bool DEMSolver::initialize() {
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
  DEMInitialize_device(getDomainOrigin(), getDomainSize(), getGPUThreadsPerBlock());

  std::cout << "DEM solver: initialization completed." << std::endl;
  return true;
}

void DEMSolver::solve() {
  removeVtuFiles(dir);
  removeDatFiles(dir);

  const double timeStep = getTimeStep();

  if (initialize()) {
    DEMNeighborSearch(getGPUThreadsPerBlock());
    handleDEMHostArray();
    outputSolidParticleVTU(dir, iFrame, iStep, timeStep);
    if(triangleWalls().hostSize() > 0) outputTriangleWallVTU(dir, iFrame, iStep, timeStep);
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
        std::cout << "DEM solver: frame " << iFrame << " at time " << iStep * timeStep
                  << std::endl;
        outputSolidParticleVTU(dir, iFrame, iStep, timeStep);
        if(triangleWalls().hostSize() > 0) outputTriangleWallVTU(dir, iFrame, iStep, timeStep);
      }
    }
  }
}