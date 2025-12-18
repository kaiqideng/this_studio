#include "kernel/DEMBaseSolver.h"

class problem:
    public DEMBaseSolver
{
public:
    double3 F_c = make_double3(0, 0, 100.e3);
    double r = 0.2;
    double c_d = 0.1;

    problem(): DEMBaseSolver(0) {}
    
    bool handleHostArrayInLoop() override
    {
        if(getStep() ==0)
        {
            getBallHandler().addBondedObjects(getBallHandler().getBallInteractions().objectPointedVector(), 
            getBallHandler().getBallInteractions().objectPointingVector(),0);
        }

        std::vector<double3> v = getBallHandler().getBalls().velocityVector();
        std::vector<double3> w = getBallHandler().getBalls().angularVelocityVector();
        std::vector<double3> f = getBallHandler().getBalls().forceVector();
        std::vector<double3> t = getBallHandler().getBalls().torqueVector();
        std::vector<double3> F_e(v.size(),make_double3(0, 0, 0));
        std::vector<double3> T_e(v.size(),make_double3(0, 0, 0));
        double time = getTime();
        double multiplier = time;
        if(multiplier > 1.0) multiplier = 1.0;
        f.back() += F_c * multiplier;
        F_e.back() += F_c * multiplier;
        for(size_t i = 0; i < v.size(); i++)
        {
            F_e[i] -= c_d * length(f[i]) * normalize(v[i]);
            T_e[i] -= c_d * length(t[i]) * normalize(w[i]);
        }
        getBallHandler().addExternalForce(F_e, 0);
        getBallHandler().addExternalTorque(T_e, 0);
        return true;
    }
};

int main()
{
    problem test;
    test.setProblemName("11BondedBallsBeam");

    std::vector<double3> positions(1,make_double3(0, 0, 0));
    std::vector<double> radius(1,test.r);
    test.addFixedCluster(positions, radius, 0);

    positions.resize(10,make_double3(2 * test.r, 0, 0));
    for(size_t i = 0; i < 10; i++)
    {
        positions[i].x = 2 * test.r * (i + 1);
    }
    radius.resize(10,test.r);
    std::vector<double3> velocities(10,make_double3(0, 0, 0));
    std::vector<double3> angularVelocities(10,make_double3(0, 0, 0));
    test.addCluster(positions, velocities, angularVelocities, radius, 7800, 0);

    test.setBondedContactModelForPair(0, 0, 1.0, 200e9, 2.6, 0.0, 0.0, 0.0);

    test.setTimeStep(1.e-5);
    test.setDomain(make_double3(-test.r, -test.r, -test.r), make_double3(22 * test.r, 2 * test.r, 2 * test.r));

    test.setMaximumTime(5.0);
    test.setNumFrames(10);

    test.solve();
}