#pragma once
#include "myStruct/interaction.h"
#include "myStruct/particle.h"
#include "myStruct/spatialGrid.h"
#include "ballNeighborSearch.h"

class ballHandler
{
public:
    ballHandler(cudaStream_t s)
    {
        stream_ = s;
        downLoadFlag_ = false;
        addBondedInteractionsFlag_ = false;
    }

    ~ballHandler() = default;

    void addCluster(std::vector<double3> positions, 
    std::vector<double3> velocities, 
    std::vector<double3> angularVelocities, 
    std::vector<double> radius, 
    double density, 
    int materialID)
    {
        if(!downLoadFlag_) 
        {
            downLoadFlag_ = true;
            balls_.upload(stream_);
        }
        for (size_t i = 0; i < positions.size(); i++)
        {
            double mass = 4.0 / 3.0 * pi() * pow(radius[i], 3.0) * density;
            double invMass = 0.0;
            if(mass > 1.e-20) invMass = 1. / mass;
            balls_.addHost(positions[i], 
            velocities[i], 
            angularVelocities[i], 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            radius[i], 
            invMass, 
            materialID, 
            -1);
        }
    }

    void addFixedCluster(std::vector<double3> positions, 
    std::vector<double> radius, 
    int materialID)
    {
        if(!downLoadFlag_) 
        {
            downLoadFlag_ = true;
            balls_.upload(stream_);
        }
        for (size_t i = 0; i < positions.size(); i++)
        {
            balls_.addHost(positions[i], 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            make_double3(0, 0, 0), 
            radius[i], 
            0.0, 
            materialID, 
            -1);
        }
    }

    void addBondedballInteraction(int object0, int object1)
    {
        bondedBallIndices0_.push_back(object0);
        bondedBallIndices1_.push_back(object1);
        addBondedInteractionsFlag_ = true;
    }

protected:
    ball &balls() {return balls_;}

    solidInteraction &ballInteractions() {return ballInteractions_;}

    bondedInteraction &bondedBallInteractions() {return bondedBallInteractions_;}

    void addBallExternalForce(const std::vector<double3> externalForce)
    {
        if(externalForce.size() > balls_.hostSize()) return;
        std::vector<double3> force = balls_.forceVector();
        std::vector<double3> totalF(balls_.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(
        externalForce.begin(), externalForce.end(),
        force.begin(), 
        totalF.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        balls_.setForceVector(totalF, stream_);
    }

    void addBallExternalTorque(const std::vector<double3> externalTorque)
    {
        if(externalTorque.size() > balls_.hostSize()) return;
        std::vector<double3> torque = balls_.torqueVector();
        std::vector<double3> totalT(balls_.hostSize(),make_double3(0.0, 0.0, 0.0));
        
        std::transform(
        externalTorque.begin(), externalTorque.end(),
        torque.begin(), 
        totalT.begin(),
        [](const double3& elem_a, const double3& elem_b) {return elem_a + elem_b;});
        balls_.setTorqueVector(totalT, stream_);
    }

    void downLoadNewBondedballInteractions()
    {
        if(addBondedInteractionsFlag_)
        {
            bondedBallInteractions_.add(bondedBallIndices0_,
            bondedBallIndices1_,
            balls_.positionVector(),
            stream_);
            bondedBallIndices0_.clear();
            bondedBallIndices1_.clear();
            addBondedInteractionsFlag_ = true;
        }
    }

    void ballInitialize(const double3 domainOrigin, const double3 domainSize)
    {
        downLoadNewBalls();
        double factor = 1.0;
        if(bondedBallInteractions_.deviceSize() > 0) factor = 1.1;
        double cellSizeOneDim = 0.0;
        std::vector<double> rad = balls_.radiusVector();
        if(rad.size() > 0) cellSizeOneDim = *std::max_element(rad.begin(), rad.end()) * 2.0 * factor;
        if(cellSizeOneDim > spatialGrids_.cellSize.x 
        || cellSizeOneDim > spatialGrids_.cellSize.y 
        || cellSizeOneDim > spatialGrids_.cellSize.z)
        {
            spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream_);
        }
    }

    void ballNeighborSearch(const size_t maxThreadsPerBlock);

    void ball1stHalfIntegration(const double3 g, const double t, const size_t maxThreadsPerBlock);

    void ballContactCalculation(contactModelParameters &contactModelParams, const double t, const size_t maxThreadsPerBlock);

    void ball2ndHalfIntegration(const double3 g, const double t, const size_t maxThreadsPerBlock);

    void outputBallVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time);

    virtual bool handelBallHostArray(){return false;}

private:
    void downLoadNewBalls()
    {
        if(downLoadFlag_)
        {
            balls_.download(stream_);
            ballInteractions_.alloc(balls_.deviceSize() * 6, stream_);
            ballInteractionMap_.alloc(balls_.deviceSize(), balls_.deviceSize(), stream_);
            downLoadFlag_ = false;
        }
    }

    cudaStream_t stream_;
    bool downLoadFlag_;
    bool addBondedInteractionsFlag_;
    std::vector<int> bondedBallIndices0_;
    std::vector<int> bondedBallIndices1_;

    ball balls_;
    solidInteraction ballInteractions_;
    bondedInteraction bondedBallInteractions_;

    spatialGrid spatialGrids_;
    interactionMap ballInteractionMap_;
};