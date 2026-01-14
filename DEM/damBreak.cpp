#include "kernel/WCSPHSolver.h"

inline std::vector<double3> getRegularPackedPoints(double3 origin, double3 size, double spacing)
{
    int numPX = int((size.x + 1.e-10) / spacing);
    int numPY = int((size.y + 1.e-10) / spacing);
    int numPZ = int((size.z + 1.e-10) / spacing);
    if (numPX == 0 || numPY == 0 || numPZ == 0)
    {
        std::vector<double3> p(1, origin + 0.5 * size);
        return p;
    }
    double dx = size.x / double(numPX);
    double dy = size.y / double(numPY);
    double dz = size.z / double(numPZ);
    std::vector<double3> positions;
    for (int x = 1; x <= numPX; x++)
    {
        for (int y = 1; y <= numPY; y++)
        {
            for (int z = 1; z <= numPZ; z++)
            {
                double3 pos = make_double3(0, 0, 0);
                pos.x = origin.x + dx * (double(x) - 0.5);
                pos.y = origin.y + dy * (double(y) - 0.5);
                pos.z = origin.z + dz * (double(z) - 0.5);
                positions.push_back(pos);
            }
        }
    }
    return positions;
}

class problem:
    public WCSPHSolver
{
public:
    problem(): WCSPHSolver(0) {}

    double H = 0.1;
    double spacing = 0.01;
};

int main()
{
    problem test;
    test.setProblemName("damBreak");
    std::vector<double3> p_SPH = getRegularPackedPoints(make_double3(0,0,0), 
    make_double3(test.H,2 * test.H,test.H), 
    test.spacing);
    double3 thick_w = make_double3(3 * test.spacing, 3 * test.spacing, 3 * test.spacing);
    std::vector<double3> p0_dumy = getRegularPackedPoints(make_double3(0,0,0) - thick_w, 
    make_double3(5 * test.H,2 * test.H,2 * test.H) + 2 * thick_w, 
    test.spacing);
    std::vector<double3> p_dummy;
    for(size_t i = 0; i < p0_dumy.size(); i++)
    {
        if (p0_dumy[i].x < 0 || p0_dumy[i].y < 0 || p0_dumy[i].z < 0 || p0_dumy[i].x > 5 * test.H || p0_dumy[i].y > 2 * test.H) p_dummy.push_back(p0_dumy[i]);
    }

    double c = 20 * std::sqrt(9.81 * test.H);
    test.addWCSPHDummyParticles(p_dummy, make_double3(0, 0, 0), c, test.spacing, 1000);
    test.addWCSPHParticles(p_SPH, make_double3(0, 0, 0), c, test.spacing, 1000, 1.e-3);
    
    test.setDomain(make_double3(0,0,0) - thick_w, make_double3(5 * test.H, 2 * test.H,2 * test.H) + 2 * thick_w);
    test.setGravity(make_double3(0.0, 0.0, -9.81));
    test.setTimeStep(0.25 * 1.3 * test.spacing / (c + 2.0 * std::sqrt(9.81 * test.H)));
    test.setMaximumTime(5.0);
    test.setNumFrames(100);
    test.solve();
}