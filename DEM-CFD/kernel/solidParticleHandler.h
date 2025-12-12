#pragma once
#include "myContainer/myParticle.h"
#include "myContainer/myInteraction.h"
#include "myContainer/mySpatialGrid.h"
#include "myContainer/myContactModelParams.h"
#include "myContainer/myUtility/myMat.h"
#include "myContainer/myUtility/myFileEdit.h"
#include "DEMIntegration.h"
#include "neighborSearch.h"

class solidParticleHandler
{
public:
    solidParticleHandler(cudaStream_t s)
    {
        solidParticleStream_ = s;
        solidParticlesHostArrayChangedFlag_ = false;
        clumpsHostArrayChangedFlag_ = false;
        bondedSolidParticleInteractionsHostArrayChangedFlag_ = false;
    }

    ~solidParticleHandler() = default;

    void addCluster(std::vector<double3> positions, std::vector<double3> velocities, std::vector<double3> angularVelocities, std::vector<double> radius, 
        double density, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag_)
        {
            solidParticles_.upload(solidParticleStream_);
            solidParticlesHostArrayChangedFlag_ = true;
        }
        
        for (size_t i = 0; i < positions.size(); i++)
        {
            double mass = 4.0 / 3.0 * pi() * pow(radius[i], 3.0) * density;
            double invMass = 0.0;
            if(mass > 1.e-20) invMass = 1. / mass;
            solidParticles_.add(positions[i], velocities[i], radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), angularVelocities[i], 
            invMass, materialID, -1);
        }
    }

    void addFixedCluster(std::vector<double3> positions, std::vector<double> radius, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag_)
        {
            solidParticles_.upload(solidParticleStream_);
            solidParticlesHostArrayChangedFlag_ = true;
        }

        for (size_t i = 0; i < positions.size(); i++)
        {
            solidParticles_.add(positions[i], make_double3(0.0, 0.0, 0.0), radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
            0.0, materialID, -1);
        }
    }

    void addClump(std::vector<double3> points, std::vector<double> radius, 
        double3 centroidPosition, double3 velocity, double3 angularVelocity, double mass, 
        symMatrix inertiaTensor, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag_)
        {
            solidParticles_.upload(solidParticleStream_);
            solidParticlesHostArrayChangedFlag_ = true;
        }
        if(!clumpsHostArrayChangedFlag_)
        {
            clumps_.upload(solidParticleStream_);
            clumpsHostArrayChangedFlag_ = true;
        }

        int clumpID = clumps_.hostSize();
        size_t pebbleStart = solidParticles_.hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        double volume = 0;
        for (size_t i = 0; i < points.size(); i++)
        {
            volume += 4.0 / 3.0 * pi() * pow(radius[i], 3.0);
        }
        double density_ave = 0;
        if (volume > 0.) density_ave = mass / volume;
        for (size_t i = 0; i < points.size(); i++)
        {
            double mass_i = 4.0 / 3.0 * pi() * pow(radius[i], 3.0) * density_ave;
            double invMass_i = 0.0;
            if(mass_i > 1.e-20) invMass_i = 1.0 / mass_i;
            solidParticles_.add(points[i], velocity + cross(angularVelocity, points[i] - centroidPosition), radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), angularVelocity, 
            invMass_i, materialID, clumpID);
        }

        double invMass = 0.0;
        if(mass > 1.e-20) invMass = 1.0 / mass;
        clumps_.add(centroidPosition, velocity, angularVelocity, 
        make_quaternion(1.0,0.0,0.0,0.0), inverse(inertiaTensor), invMass, 
        pebbleStart, pebbleEnd);
    }

    void addFixedClump(std::vector<double3> points, std::vector<double> radius, double3 centroidPosition, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag_)
        {
            solidParticles_.upload(solidParticleStream_);
            solidParticlesHostArrayChangedFlag_ = true;
        }
        if(!clumpsHostArrayChangedFlag_)
        {
            clumps_.upload(solidParticleStream_);
            clumpsHostArrayChangedFlag_ = true;
        }

        int clumpID = clumps_.hostSize();
        size_t pebbleStart = solidParticles_.hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        for (size_t i = 0; i < points.size(); i++)
        {
            solidParticles_.add(points[i], make_double3(0.0, 0.0, 0.0), radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
            0.0, materialID, clumpID);
        }

        clumps_.add(centroidPosition, make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0,0.0,0.0,0.0), make_symMatrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 
        pebbleStart, pebbleEnd);
    }

    void addBondedSolidParticleInteractions(int object0, int object1)
    {
        // Do not need to upload because the bondedInteraction host array is not changed in this function
        bondedObjects0_.push_back(object0);
        bondedObjects1_.push_back(object1);

        bondedSolidParticleInteractionsHostArrayChangedFlag_ = true;
    }

protected:
    void addSolidParticleExternalForce(const std::vector<double3> externalForce)
    {
        if(externalForce.size() > solidParticles_.hostSize()) return;
        std::vector<double3> force = solidParticles_.getForceVectors();
        std::vector<double3> totalForce(solidParticles_.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(
        externalForce.begin(), externalForce.end(),
        force.begin(), 
        totalForce.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        solidParticles_.setForceVectors(totalForce);
    }

    void addSolidParticleExternalTorque(const std::vector<double3> externalTorque)
    {
        if(externalTorque.size() > solidParticles_.hostSize()) return;
        std::vector<double3> torque = solidParticles_.getTorqueVectors();
        std::vector<double3> totalTorque(solidParticles_.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(
        externalTorque.begin(), externalTorque.end(),
        torque.begin(),
        totalTorque.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        solidParticles_.setTorqueVectors(totalTorque);
    }

    solidParticle &solidParticles() {return solidParticles_;}

    const std::vector<double3> getSolidParticlePosition() {return solidParticles_.getPositionVectors();}

    const std::vector<double3> getSolidParticleVelocity() {return solidParticles_.getVelocityVectors();}

    const std::vector<double3> getSolidParticleAngularVelocity() {return solidParticles_.getAngularVelocityVectors();}

    const std::vector<double3> getSolidParticleForce() {return solidParticles_.getForceVectors();}

    const std::vector<double3> getSolidParticleTorque() {return solidParticles_.getTorqueVectors();}

    const std::vector<int> getSolidParticleInteractionObjectPointed() {return solidParticleInteractions_.getObjectPointedVectors();}

    const std::vector<int> getSolidParticleInteractionObjectPointing() {return solidParticleInteractions_.getObjectPointingVectors();}

    void solidParticleInitialize(const double3 domainOrigin, const double3 domainSize)
    {
        downLoadSolidParticlesAndInteractions();
        downloadSolidParticleSpatialGrids(domainOrigin, domainSize);
    }

    void solidParticleNeighborSearch(const size_t maxThreadsPerBlock)
    {
        launchSolidParticleNeighborSearch(solidParticleInteractions_, solidParticles_, solidParticleSpatialGrids_, maxThreadsPerBlock, solidParticleStream_);
    }

    void solidParticleIntegrateBeforeContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchSolidParticleIntegrateBeforeContact(solidParticles_, clumps_, gravity, timeStep, maxThreadsPerBlock, solidParticleStream_);
        cudaDeviceSynchronize();
        solidParticles_.clearForceTorque(solidParticleStream_);
        clumps_.clearForceTorque(solidParticleStream_);
    }

    void solidParticleInteractionCalculation(solidContactModelParameter& solidContactModelParameters, const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchSolidParticleInteractionCalculation(solidParticleInteractions_, bondedSolidParticleInteractions_, solidParticles_, clumps_, 
        solidContactModelParameters, timeStep, maxThreadsPerBlock, solidParticleStream_);
    }

    void solidParticleIntegrateAfterContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchSolidParticleIntegrateAfterContact(solidParticles_, clumps_, gravity, timeStep, maxThreadsPerBlock, solidParticleStream_);
    }

    void outputSolidParticleVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double timeStep);

private:
    void downLoadSolidParticlesAndInteractions()
    {
        if(solidParticlesHostArrayChangedFlag_)
        {
            solidParticles_.download(solidParticleStream_);
            solidParticleInteractions_.current.allocDeviceArray(6 * solidParticles_.hostSize(), solidParticleStream_);
            solidParticlesHostArrayChangedFlag_ = false;
        }
        if(clumpsHostArrayChangedFlag_)
        {
            clumps_.download(solidParticleStream_);
            clumpsHostArrayChangedFlag_ = false;
        }
        if(bondedSolidParticleInteractionsHostArrayChangedFlag_)
        {
            if(solidParticles_.hostSize() == 0) return;
            bondedSolidParticleInteractions_.add(bondedObjects0_, bondedObjects1_, getSolidParticlePosition(), solidParticleStream_);
            bondedObjects0_.clear();
            bondedObjects1_.clear();
            bondedSolidParticleInteractionsHostArrayChangedFlag_ = false;
        }
    }

    void downloadSolidParticleSpatialGrids(double3 domainOrigin, double3 domainSize)
    {
        double cellSizeOneDim = 0.0;
        const std::vector<double> radii = solidParticles_.getEffectiveRadii();
        if(radii.size() > 0) cellSizeOneDim = *std::max_element(radii.begin(), radii.end()) * 2.0;
        double3 cellSize = solidParticleSpatialGrids_.getCellSize();
        if(cellSizeOneDim > cellSize.x || cellSizeOneDim > cellSize.y || cellSizeOneDim > cellSize.z)
        {
            solidParticleSpatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, solidParticleStream_);
        }
    }

    cudaStream_t solidParticleStream_;
    
    bool solidParticlesHostArrayChangedFlag_;
    bool clumpsHostArrayChangedFlag_;
    bool bondedSolidParticleInteractionsHostArrayChangedFlag_;
    std::vector<int> bondedObjects0_;
    std::vector<int> bondedObjects1_;

    solidParticle solidParticles_;
    clump clumps_;
    spatialGrid solidParticleSpatialGrids_;
    interactionSpringSystem solidParticleInteractions_;
    interactionBonded bondedSolidParticleInteractions_;
};