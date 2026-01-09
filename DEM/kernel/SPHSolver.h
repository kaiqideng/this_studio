#pragma once
#include "solverParams.h"
#include "SPHHandler.h"

class SPHSolver:
    public solverParams
{
public:
    SPHSolver(cudaStream_t s) : solverParams()
    {
		setNameFlag_ = false;
		dir_ = "SPHOutput";
		iStep_ = 0;
		iFrame_ = 0;
		time_ = 0.0;

		stream_ = s;
		SPHGPUGridDim_ = 1;
		SPHGPUBlockDim_ = 1;
		ghostGPUGridDim_ = 1;
		ghostGPUBlockDim_ = 1;
	}

	~SPHSolver() = default;

	void setProblemName(const std::string& name) 
	{
		if(!setNameFlag_) dir_ = name; 
		setNameFlag_ = true;
	}

	void addSPHParticles(std::vector<double3> points, double3 velocity, double spacing, double density, double kinematicViscosity)
    {
        SPHHandler_.addSPHParticles(points, velocity, spacing, density, kinematicViscosity, stream_);
    }

	void addGhostParticles(std::vector<double3> points, double3 velocity, double spacing, double density)
    {
        SPHHandler_.addGhostParticles(points, velocity, spacing, density, stream_);
    }

	void solve()
	{
		removeVtuFiles(dir_);
		removeDatFiles(dir_);

		const double timeStep = getTimeStep();

		if (initialize()) 
		{
			if (handleHostArrayInLoop()) download();
			outputData();
			size_t numSteps = size_t((getMaximumTime()) / timeStep) + 1;
			size_t frameInterval = numSteps / getNumFrames();
			if (frameInterval < 1) frameInterval = 1;

			while (iStep_ <= numSteps) 
			{
				iStep_++;
				time_ += timeStep;
				neighborSearch();
				integration(timeStep);
				if(handleHostArrayInLoop()) download();
				if (iStep_ % frameInterval == 0) 
				{
					iFrame_++;
					std::cout << "DEM solver: frame " << iFrame_ << " at time " << time_ << std::endl;
					outputData();
				}
			}
		}
	}

protected:
    const std::string& getDir() const {return dir_;}

	const size_t& getStep() const {return iStep_;}

    const size_t& getFrame() const {return iFrame_;}

	const double& getTime() const {return time_;}

    virtual bool handleHostArrayInLoop() {return false;}

	SPHHandler& getSPHHandler() {return SPHHandler_;}

private:
    virtual void download() 
	{
		SPHHandler_.download(getDomainOrigin(), getDomainSize(), stream_);

		const size_t maxThreads = getGPUMaxThreadsPerBlock();
		const size_t numSPHs = SPHHandler_.getSPHAndGhosts().SPHDeviceSize();
		if (numSPHs > 0 && maxThreads > 1) SPHGPUBlockDim_ = maxThreads < numSPHs ? maxThreads : numSPHs;
		if (SPHGPUBlockDim_ > 0) SPHGPUGridDim_ = (numSPHs + SPHGPUBlockDim_ - 1) / SPHGPUBlockDim_;

        const size_t numGhosts = SPHHandler_.getSPHAndGhosts().ghostDeviceSize();
		if (numGhosts > 0 && maxThreads > 1) ghostGPUBlockDim_ = maxThreads < numGhosts ? maxThreads : numGhosts;
		if (ghostGPUBlockDim_ > 0) ghostGPUGridDim_ = (numGhosts + ghostGPUBlockDim_ - 1) / ghostGPUBlockDim_;
	}

	bool initialize()
	{
		std::cout << "SPH solver: initializing..." << std::endl;

		std::cout << "SPH solver: using GPU Device " << getGPUDeviceIndex() << std::endl;
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(getGPUDeviceIndex());
		if (cudaStatus != cudaSuccess) 
		{
			std::cout << "SPH solver: cudaSetDevice( " << getGPUDeviceIndex()
			<< " ) failed!  Do you have a CUDA-capable GPU installed?"
			<< std::endl;
			exit(1);
		}
		std::cout << "SPH solver: downloading array from host to device..." << std::endl;
		download();
		neighborSearch();
		std::cout << "DEM solver: initialization completed." << std::endl;
		return true;
	}

	virtual void outputData()
	{
		SPHHandler_.outputSPHVTU(getDir(), getFrame(), getStep(), getTime());
	}

	virtual void neighborSearch()
	{
		SPHHandler_.neighborSearch(getGPUMaxThreadsPerBlock(), stream_);
	}

	virtual void integration(const double dt)
	{
		SPHHandler_.integration1st(getGravity(), dt, SPHGPUGridDim_, SPHGPUBlockDim_, stream_);
		SPHHandler_.updateBoundaryCondition(getGravity(), dt, ghostGPUGridDim_, ghostGPUBlockDim_, stream_);
		SPHHandler_.integration2nd(dt, SPHGPUGridDim_, SPHGPUBlockDim_, stream_);
		SPHHandler_.updateBoundaryCondition(getGravity(), dt, ghostGPUGridDim_, ghostGPUBlockDim_, stream_);
		SPHHandler_.integration3rd(dt, SPHGPUGridDim_, SPHGPUBlockDim_, stream_);
	}

    bool setNameFlag_;
    std::string dir_;
    size_t iStep_;
    size_t iFrame_;;
	double time_;

	cudaStream_t stream_;

	SPHHandler SPHHandler_;
	size_t SPHGPUGridDim_;
	size_t SPHGPUBlockDim_;
	size_t ghostGPUGridDim_;
	size_t ghostGPUBlockDim_;
};