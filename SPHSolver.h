#pragma once
#include "DEM/kernel/solverParams.h"
#include "SPHHandler.h"
#include "SPHIntegration.h"
#include <cstddef>

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
		SPHGridDim_ = 1;
		SPHBlockDim_ = 1;
	}

	~SPHSolver() = default;

	void setProblemName(const std::string& name) 
	{
		if(!setNameFlag_) dir_ = name; 
		setNameFlag_ = true;
	}

	void addWCSPHParticles(std::vector<double3> points, double3 velocity, double soundSpeed, double spacing, double density, double kinematicViscosity)
    {
        SPHHandler_.addWCSPHParticles(points, velocity, soundSpeed, spacing, density, kinematicViscosity, stream_);
    }

	void addWCSPHDummyParticles(std::vector<double3> points, double3 velocity, double soundSpeed, double spacing, double density)
    {
        SPHHandler_.addWCSPHDummyParticles(points, velocity, soundSpeed, spacing, density, stream_);
    }

	void addISPHParticles(std::vector<double3> points, double3 velocity, double spacing, double density, double kinematicViscosity)
    {
        SPHHandler_.addISPHParticles(points, velocity, spacing, density, kinematicViscosity, stream_);
    }

	void addISPHGhostParticles(std::vector<double3> points, double3 velocity, double spacing, double density)
    {
        SPHHandler_.addISPHGhostParticles(points, velocity, spacing, density, stream_);
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
				if (handleHostArrayInLoop()) download();
				if (iStep_ % frameInterval == 0) 
				{
					iFrame_++;
					std::cout << "SPH solver: frame " << iFrame_ << " at time " << time_ << std::endl;
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
    void download() 
	{
		SPHHandler_.download(getDomainOrigin(), getDomainSize(), stream_);

		const size_t maxThreads = getGPUMaxThreadsPerBlock();
		const size_t numSPHs = SPHHandler_.getWCSPHs().SPHDeviceSize();
		if (numSPHs == 0) SPHHandler_.getISPHs().SPHDeviceSize();
		if (numSPHs > 0 && maxThreads > 1) SPHBlockDim_ = maxThreads < numSPHs ? maxThreads : numSPHs;
		if (SPHBlockDim_ > 0) SPHGridDim_ = (numSPHs + SPHBlockDim_ - 1) / SPHBlockDim_;
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

	void outputData()
	{
		SPHHandler_.outputSPHVTU(getDir(), getFrame(), getStep(), getTime());
	}

	virtual void neighborSearch()
	{
		SPHHandler_.WCSPHNeighborSearch(getGPUMaxThreadsPerBlock(), stream_);
		if (getStep() == 0) SPHHandler_.setDummyParticleBoundary(getGPUMaxThreadsPerBlock(), stream_);
	}

    virtual void integration(const double dt)
    {
		SPHHandler_.WCSPH1stIntegration(getDomainOrigin(), dt, SPHGridDim_, SPHBlockDim_, stream_);
		SPHHandler_.WCSPH2ndIntegration(getDomainOrigin(), dt, SPHGridDim_, SPHBlockDim_, stream_);
    }

    bool setNameFlag_;
    std::string dir_;
    size_t iStep_;
    size_t iFrame_;;
	double time_;

	cudaStream_t stream_;

	SPHHandler SPHHandler_;
	size_t SPHGridDim_;
	size_t SPHBlockDim_;
};