#include "DEMBaseSolver.h"

void DEMBaseSolver::download()
{
    downloadContactModelParams(stream_);
    ballHandler_.download(getDomainOrigin(), getDomainSize(), stream_);
}

bool DEMBaseSolver::initialize()
{
    std::cout << "DEM solver: initializing..." << std::endl;

    std::cout << "DEM solver: using GPU Device " << getGPUDeviceIndex() << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(getGPUDeviceIndex());
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "DEM solver: cudaSetDevice( " << getGPUDeviceIndex()
        << " ) failed!  Do you have a CUDA-capable GPU installed?"
        << std::endl;
        exit(1);
    }
    std::cout << "DEM solver: downloading array from host to device..." << std::endl;
    download();
    neighborSearch();
    std::cout << "DEM solver: initialization completed." << std::endl;
    return true;
}

void DEMBaseSolver::outputData()
{
    ballHandler_.outputBallVTU(getDir(), getFrame(), getStep(), getTime());
}

void DEMBaseSolver::neighborSearch()
{
    ballHandler_.neighborSearch(getGPUMaxThreadsPerBlock(), stream_);
}

void DEMBaseSolver::integration1st(const double dt)
{
    ballHandler_.integration1st(getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
}

void DEMBaseSolver::contactCalculation(const double dt)
{
    ballHandler_.contactCalculation(getContactModelParams(), dt, getGPUMaxThreadsPerBlock(), stream_);
}

void DEMBaseSolver::integration2nd(const double dt)
{
    ballHandler_.integration2nd(getGravity(), dt, getGPUMaxThreadsPerBlock(), stream_);
}

void DEMBaseSolver::solve()
{
    removeVtuFiles(dir_);
    removeDatFiles(dir_);

    const double timeStep = getTimeStep();

    if (initialize()) 
    {
        if (handleHostArrayInLoop()) {download();}
        outputData();
        size_t numSteps = size_t((getMaximumTime()) / timeStep) + 1;
        size_t frameInterval = numSteps / getNumFrames();
        if (frameInterval < 1) frameInterval = 1;

        while (iStep_ <= numSteps) 
        {
            iStep_++;
            time_ += timeStep;
            neighborSearch();
            integration1st(timeStep);
            contactCalculation(timeStep);
            if(handleHostArrayInLoop()) {download();}
            integration2nd(timeStep);
            if (iStep_ % frameInterval == 0) 
            {
                iFrame_++;
                std::cout << "DEM solver: frame " << iFrame_ << " at time " << time_ << std::endl;
                outputData();
            }
        }
    }
}

bool DEMBaseSolver::handleHostArrayInLoop() 
{
    return false;
}

ballHandler& DEMBaseSolver::getBallHandler()
{
    return ballHandler_;
}