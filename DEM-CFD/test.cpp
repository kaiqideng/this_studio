#include "kernel/DEMSolver.h"

class problem:
    public DEMSolver
{
public:
    double3 F_c = make_double3(0, 0, 100.e3);
    double r = 0.2;
    double c_d = 0.1;

    problem(): DEMSolver(0) {}
    bool handleDEMHostArray() override
    {
        if(getStep() ==0 ) return false;

        std::vector<double3> v = getSolidParticleVelocity();
        std::vector<double3> w = getSolidParticleAngularVelocity();
        std::vector<double3> f = getSolidParticleForce();
        std::vector<double3> t = getSolidParticleTorque();
        std::vector<double3> F_e(v.size(),make_double3(0, 0, 0));
        std::vector<double3> T_e(v.size(),make_double3(0, 0, 0));
        f.back() += F_c;
        F_e.back() += F_c;
        for(size_t i = 0; i < v.size(); i++)
        {
            F_e[i] -= c_d * length(f[i]) * normalize(v[i]);
            T_e[i] -= c_d * length(t[i]) * normalize(w[i]);
        }
        addSolidParticleExternalForce(F_e);
        addSolidParticleExternalTorque(T_e);
        return true;
    }
};

int main()
{
    problem test;

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

    test.setBondedModel(0, 0, 1.0, 200e9, 2.6, 0.0, 0.0, 0.0);

    test.setTimeStep(1.e-5);
    test.setDomain(make_double3(-test.r, -test.r, -test.r), make_double3(22 * test.r, 2 * test.r, 2 * test.r));

    test.setMaximumTime(5.0);
    test.setNumFrames(10);

    for(size_t i = 0; i < 10; i++)
    {
        test.addBondedSolidParticleInteractions(i, i+1);
    }

    test.solve();
}