#include "DEMBaseSolver.h"

void DEMBaseSolver::download()
{
    downloadContactModelParams(stream_);

    size_t numBalls0 = ballHandler_.balls().deviceSize();
    ballHandler_.download(getDomainOrigin(), getDomainSize(), stream_);
    size_t numBalls1 = ballHandler_.balls().deviceSize();
    if(numBalls1 > numBalls0)
    {
        ballInteractions_.alloc(numBalls1 * 6, stream_);
        ballInteractionMap_.alloc(numBalls1, numBalls1, stream_);
    }
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
    launchBallNeighborSearch(ballInteractions_, 
    ballInteractionMap_, 
    ballHandler_.balls(),
    getBallSpatialGrids(),
    getGPUMaxThreadsPerBlock(),
    stream_);
}

void DEMBaseSolver::integration1st(const double dt)
{
    launchBall1stHalfIntegration(ballHandler_.balls(), 
    getGravity(), 
    dt, 
    getGPUMaxThreadsPerBlock(), 
    stream_);
}

void DEMBaseSolver::contactCalculation(const double dt)
{
    launchBallContactCalculation(ballInteractions_, 
    bondedBallInteractions_, 
    ballHandler_.balls(), 
    getContactModelParams(), 
    ballInteractionMap_,
    dt, 
    getGPUMaxThreadsPerBlock(), 
    stream_);
}

void DEMBaseSolver::integration2nd(const double dt)
{
    launchBall2ndHalfIntegration(ballHandler_.balls(), 
    getGravity(), 
    dt, 
    getGPUMaxThreadsPerBlock(), 
    stream_);
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

void DEMBaseSolver::addBondedObjects(const std::vector<int> &object0, const std::vector<int> &object1)
{
    bondedBallInteractions_.add(object0,
    object1,
    ballHandler_.balls().positionVector(),
    stream_);
}

void DEMBaseSolver::addBallExternalForce(const std::vector<double3>& externalForce)
{
    if(externalForce.size() > ballHandler_.balls().hostSize()) return;
    std::vector<double3> force = ballHandler_.balls().forceVector();
    std::vector<double3> totalF(ballHandler_.balls().hostSize(),make_double3(0.0, 0.0, 0.0));
    
    std::transform(
    externalForce.begin(), externalForce.end(),
    force.begin(), 
    totalF.begin(),
    [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
    ballHandler_.balls().setForceVector(totalF, stream_);
}

void DEMBaseSolver::addBallExternalTorque(const std::vector<double3>& externalTorque)
{
    if(externalTorque.size() > ballHandler_.balls().hostSize()) return;
    std::vector<double3> torque = ballHandler_.balls().torqueVector();
    std::vector<double3> totalT(ballHandler_.balls().hostSize(),make_double3(0.0, 0.0, 0.0));
    
    std::transform(
    externalTorque.begin(), externalTorque.end(),
    torque.begin(), 
    totalT.begin(),
    [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
    ballHandler_.balls().setTorqueVector(totalT, stream_);
}

bool DEMBaseSolver::handleHostArrayInLoop() 
{
    return false;
}

ballHandler& DEMBaseSolver::getBallHandler()
{
    return ballHandler_;
}

spatialGrid& DEMBaseSolver::getBallSpatialGrids() 
{
    return ballHandler_.spatialGrids();
}