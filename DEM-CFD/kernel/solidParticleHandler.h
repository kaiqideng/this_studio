#pragma once
#include "myContainer/myParticle.h"
#include "myContainer/myInteraction.h"
#include "myContainer/mySpatialGrid.h"
#include "myContainer/myContactModelParams.h"
#include "myContainer/myUtility/myMat.h"
#include "myContainer/myUtility/myFileEdit.h"
#include "integration.h"
#include "neighborSearch.h"

class solidParticleHandler
{
public:
    solidParticleHandler(cudaStream_t s)
    {
        solidParticleStream = s;
        solidParticlesHostArrayChangedFlag = false;
        clumpsHostArrayChangedFlag = false;
        bondedSolidParticleInteractionsHostArrayChangedFlag = false;
    }

    ~solidParticleHandler() = default;

    void addCluster(std::vector<double3> positions, std::vector<double3> velocities, std::vector<double3> angularVelocities, std::vector<double> radius, 
        double density, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag)
        {
            solidParticles.upload(solidParticleStream);
            solidParticlesHostArrayChangedFlag = true;
        }
        
        for (size_t i = 0; i < positions.size(); i++)
        {
            double mass = 4.0 / 3.0 * pi() * pow(radius[i], 3.0) * density;
            double invMass = 0.0;
            if(mass > 1.e-20) invMass = 1. / mass;
            solidParticles.add(positions[i], velocities[i], radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), angularVelocities[i], 
            invMass, materialID, -1);
        }
    }

    void addFixedCluster(std::vector<double3> positions, std::vector<double> radius, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag)
        {
            solidParticles.upload(solidParticleStream);
            solidParticlesHostArrayChangedFlag = true;
        }

        for (size_t i = 0; i < positions.size(); i++)
        {
            solidParticles.add(positions[i], make_double3(0.0, 0.0, 0.0), radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
            0.0, materialID, -1);
        }
    }

    void addClump(std::vector<double3> points, std::vector<double> radius, 
        double3 centroidPosition, double3 velocity, double3 angularVelocity, double mass, 
        symMatrix inertiaTensor, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag)
        {
            solidParticles.upload(solidParticleStream);
            solidParticlesHostArrayChangedFlag = true;
        }
        if(!clumpsHostArrayChangedFlag)
        {
            clumps.upload(solidParticleStream);
            clumpsHostArrayChangedFlag = true;
        }

        int clumpID = clumps.hostSize();
        size_t pebbleStart = solidParticles.hostSize();
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
            solidParticles.add(points[i], velocity + cross(angularVelocity, points[i] - centroidPosition), radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), angularVelocity, 
            invMass_i, materialID, clumpID);
        }

        double invMass = 0.0;
        if(mass > 1.e-20) invMass = 1.0 / mass;
        clumps.add(centroidPosition, velocity, angularVelocity, 
        make_quaternion(1.0,0.0,0.0,0.0), inverse(inertiaTensor), invMass, 
        pebbleStart, pebbleEnd);
    }

    void addFixedClump(std::vector<double3> points, std::vector<double> radius, double3 centroidPosition, int materialID)
    {
        if(!solidParticlesHostArrayChangedFlag)
        {
            solidParticles.upload(solidParticleStream);
            solidParticlesHostArrayChangedFlag = true;
        }
        if(!clumpsHostArrayChangedFlag)
        {
            clumps.upload(solidParticleStream);
            clumpsHostArrayChangedFlag = true;
        }

        int clumpID = clumps.hostSize();
        size_t pebbleStart = solidParticles.hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        for (size_t i = 0; i < points.size(); i++)
        {
            solidParticles.add(points[i], make_double3(0.0, 0.0, 0.0), radius[i], 
            make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
            0.0, materialID, clumpID);
        }

        clumps.add(centroidPosition, make_double3(0.0, 0.0, 0.0), make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0,0.0,0.0,0.0), make_symMatrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 
        pebbleStart, pebbleEnd);
    }

    void addBondedSolidParticleInteractions(int object0, int object1)
    {
        // Do not need to upload at first because the bondedInteraction host array is not changed in this function
        bondedObjects0.push_back(object0);
        bondedObjects1.push_back(object1);

        bondedSolidParticleInteractionsHostArrayChangedFlag = true;
    }

protected:
    void addSolidParticleExternalForce(const std::vector<double3> externalForce)
    {
        if(externalForce.size() > solidParticles.hostSize()) return;
        std::vector<double3> force = solidParticles.getForceVectors();
        std::vector<double3> totalForce(solidParticles.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(
        externalForce.begin(), externalForce.end(),
        force.begin(), 
        totalForce.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        solidParticles.setForceVectors(totalForce);
    }

    void addSolidParticleExternalTorque(const std::vector<double3> externalTorque)
    {
        if(externalTorque.size() > solidParticles.hostSize()) return;
        std::vector<double3> torque = solidParticles.getTorqueVectors();
        std::vector<double3> totalTorque(solidParticles.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(
        externalTorque.begin(), externalTorque.end(),
        torque.begin(),
        totalTorque.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        solidParticles.setTorqueVectors(totalTorque);
    }

    const std::vector<double3> getSolidParticlePosition() {return solidParticles.getPositionVectors();}

    const std::vector<double3> getSolidParticleVelocity() {return solidParticles.getVelocityVectors();}

    const std::vector<double3> getSolidParticleAngularVelocity() {return solidParticles.getAngularVelocityVectors();}

    const std::vector<double3> getSolidParticleForce() {return solidParticles.getForceVectors();}

    const std::vector<double3> getSolidParticleTorque() {return solidParticles.getTorqueVectors();}

    const std::vector<int> getSolidParticleInteractionObjectPointed() {return solidParticleInteractions.getObjectPointedVectors();}

    const std::vector<int> getSolidParticleInteractionObjectPointing() {return solidParticleInteractions.getObjectPointingVectors();}

    void solidParticleInitialize(const double3 domainOrigin, const double3 domainSize)
    {
        downLoadSolidParticlesInteractions();
        downloadSolidParticleSpatialGrids(domainOrigin, domainSize);
    }

    void solidParticleNeighborSearch(const size_t maxThreadsPerBlock)
    {
        launchSolidParticleNeighborSearch(solidParticleInteractions, solidParticles, solidParticleSpatialGrids, maxThreadsPerBlock, solidParticleStream);
    }

    void solidParticleIntegrateBeforeContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchSolidParticleIntegrateBeforeContact(solidParticles, clumps, gravity, timeStep, maxThreadsPerBlock, solidParticleStream);
    }

    void solidParticleInteractionCalculation(solidContactModelParameter& solidContactModelParameters, const double timeStep, const size_t maxThreadsPerBlock)
    {
        solidParticles.clearForceTorque(solidParticleStream);
        clumps.clearForceTorque(solidParticleStream);
        cudaDeviceSynchronize();
        launchSolidParticleInteractionCalculation(solidParticleInteractions, bondedSolidParticleInteractions, solidParticles, clumps, 
        solidContactModelParameters, timeStep, maxThreadsPerBlock, solidParticleStream);
    }

    void solidParticleIntegrateAfterContact(const double3 gravity, const double timeStep, const size_t maxThreadsPerBlock)
    {
        launchSolidParticleIntegrateAfterContact(solidParticles, clumps, gravity, timeStep, maxThreadsPerBlock, solidParticleStream);
    }

    void outputSolidParticleVTU(const std::string dir, const size_t iFrame, const size_t iStep, const double timeStep);

private:
    void downLoadSolidParticlesInteractions()
    {
        if(solidParticlesHostArrayChangedFlag)
        {
            solidParticles.download(solidParticleStream);
            solidParticleInteractions.current.allocDeviceArray(6 * solidParticles.hostSize(), solidParticleStream);
            solidParticlesHostArrayChangedFlag = false;
        }
        if(clumpsHostArrayChangedFlag)
        {
            clumps.download(solidParticleStream);
            clumpsHostArrayChangedFlag = false;
        }
        if(bondedSolidParticleInteractionsHostArrayChangedFlag)
        {
            if(solidParticles.hostSize() == 0) return;
            bondedSolidParticleInteractions.add(bondedObjects0, bondedObjects1, getSolidParticlePosition(), solidParticleStream);
            bondedObjects0.clear();
            bondedObjects1.clear();
            bondedSolidParticleInteractionsHostArrayChangedFlag = false;
        }
    }

    void downloadSolidParticleSpatialGrids(double3 domainOrigin, double3 domainSize)
    {
        double cellSizeOneDim = 0.0;
        std::vector<double> radii = solidParticles.getEffectiveRadii();
        if(radii.size() > 0) cellSizeOneDim = *std::max_element(radii.begin(), radii.end()) * 2.0;
        double3 cellSize = solidParticleSpatialGrids.getCellSize();
        if(cellSizeOneDim > cellSize.x || cellSizeOneDim > cellSize.y || cellSizeOneDim > cellSize.z)
        {
            solidParticleSpatialGrids.set(domainOrigin, domainSize, cellSizeOneDim, solidParticleStream);
        }
    }

    cudaStream_t solidParticleStream;
    
    bool solidParticlesHostArrayChangedFlag;
    bool clumpsHostArrayChangedFlag;
    bool bondedSolidParticleInteractionsHostArrayChangedFlag;
    std::vector<int> bondedObjects0;
    std::vector<int> bondedObjects1;

    solidParticle solidParticles;
    clump clumps;
    spatialGrid solidParticleSpatialGrids;
    interactionSpringSystem solidParticleInteractions;
    interactionBonded bondedSolidParticleInteractions;
};