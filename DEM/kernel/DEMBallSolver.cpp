#include "DEMBallSolver.h"
#include "myStruct/myUtility/myFileEdit.h"

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
  handelBallHostArray();
  downloadContactModelParameters();
  downLoadNewBondedballInteractions();
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
    numSteps_ = size_t((getMaximumTime()) / timeStep) + 1;
    frameInterval_ = numSteps_ / getNumFrames();
    if (frameInterval_ < 1)
      frameInterval_ = 1;

    while (iStep_ <= numSteps_) {
      iStep_++;
      time_ += timeStep;
      ballNeighborSearch(getGPUThreadsPerBlock());
      ball1stHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
      ballContactCalculation(contactModelParams_, timeStep,getGPUThreadsPerBlock());
      if(handelBallHostArray()){
        ballInitialize(getDomainOrigin(), getDomainSize());
        downloadContactModelParameters();
        downLoadNewBondedballInteractions();}
      ball2ndHalfIntegration(getGravity(),timeStep,getGPUThreadsPerBlock());
      if (iStep_ % frameInterval_ == 0) {
        iFrame_++;
        std::cout << "DEM solver: frame " << iFrame_ << " at time " << time_
                  << std::endl;
        outputBallVTU(dir_, iFrame_, iStep_, time_);
      }
    }
  }
}