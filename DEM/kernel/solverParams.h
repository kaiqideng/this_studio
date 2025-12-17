#pragma once
#include <cmath>
#include <iostream>
#include <vector_functions.h>

class solverParams
{
public:
    solverParams()
    {
        simParams.maximumTime = 1.0;
        simParams.timeStep = 1.0;
        simParams.numFrames = 1;

        GPUParams.deviceIndex = 0;
        GPUParams.maxThreadsPerBlock = 256;

        domainOrigin = make_double3(0.0, 0.0, 0.0);
        domainSize = make_double3(1.0, 1.0, 1.0);
        gravity = make_double3(0.0, 0.0, 0.0);
    }

    ~solverParams() = default;

    void setGPUDeviceIndex(size_t index) { GPUParams.deviceIndex = index; }

    void setGPUMaxThreadsPerBlock(size_t threads_max) { GPUParams.maxThreadsPerBlock = threads_max; }

    void setMaximumTime(double t)
    {
        if (t <= 0.)
        {
			std::cout << "Error: maximum time must be positive." << std::endl;
			return;
        }

        simParams.maximumTime = t;
    }

    void setNumFrames(size_t n)
    {
		if (n < 1)
		{
			std::cout << "Error: number of frames must be at least 1." << std::endl;
			return;
		}

        simParams.numFrames = n;
    }

    void setTimeStep(double dt)
    {
        if (dt <= 0.)
        {
			std::cout << "Error: time step must be positive." << std::endl;
			return;
        }

		simParams.timeStep = dt;
    }

    void setDomain(double3 origin, double3 size)
    {
        domainOrigin = origin;
        domainSize = size;
		if (size.x <= 0. || size.y <= 0. || size.z <= 0.)
		{
			std::cout << "Error: domain size must be positive in all directions." << std::endl;
			return;
		}
    }

    void setGravity(double3 g) { gravity = g; }
    
protected:
    const double &getMaximumTime() const {return simParams.maximumTime;}

    const double &getTimeStep() const {return simParams.timeStep;}

    const size_t &getNumFrames() const {return simParams.numFrames;}

    const size_t &getGPUDeviceIndex() const {return GPUParams.deviceIndex;}

    const size_t &getGPUMaxThreadsPerBlock() const {return GPUParams.maxThreadsPerBlock;}

    const double3 &getDomainOrigin() const {return domainOrigin;}

    const double3 &getDomainSize() const {return domainSize;}

    const double3 &getGravity() const {return gravity;}
    
private:
    struct simulationParameter
    {
        double maximumTime;
        double timeStep;
        size_t numFrames;
    };

    struct GPUParameter
    {
        size_t deviceIndex;
        size_t maxThreadsPerBlock;
    };

    simulationParameter simParams;
    GPUParameter GPUParams;
    double3 domainOrigin;
    double3 domainSize;
    double3 gravity;
};