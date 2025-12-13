#include "kernel/DEMBallSolver.h"
#include "myStruct/interaction.h"
#include <vector>

class problem:
    public DEMBallSolver
{
public:
    double3 F_c = make_double3(0, 0, 100.e3);
    double r = 0.2;
    double c_d = 0.1;

    problem(): DEMBallSolver(0) {}
    bool handelBallHostArray() override
    {
        if(getStep() ==0 )
        {
            const std::vector<int> ob0 = bondedBallInteractions().objectPointedVector();
            const std::vector<int> ob1 = bondedBallInteractions().objectPointingVector();
            for(size_t i = 0; i < ob0.size(); i++)
            {
                addBondedballInteraction(ob0[i], ob1[i]);
            }
        }

        std::vector<double3> v = balls().velocityVector();
        std::vector<double3> w = balls().angularVelocityVector();
        std::vector<double3> f = balls().forceVector();
        std::vector<double3> t = balls().torqueVector();
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
        addBallExternalForce(F_e);
        addBallExternalTorque(T_e);
        return true;
    }
};

int main()
{
    problem test;
    test.setProblemName("11BondedParticleBeamBendedByConstantVerticalForceAtEndOfBeam");

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