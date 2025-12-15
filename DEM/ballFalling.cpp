#include "kernel/DEMBallMeshWallSolver.h"
#include <vector_functions.h>

class problem:
    public DEMBallMeshWallSolver
{
public:
    problem(): DEMBallMeshWallSolver(0) {}

    bool handleHostArrayInLoop() override
    {
        return false;
    }
};

int main()
{
    problem test;
    test.setProblemName("ballFalling");

    std::vector<double3> positions(1,make_double3(0.1, 0.1, 0.0));
    std::vector<double> radius(1,0.01);
    std::vector<double3> velocities(1,make_double3(0.0, 0.0, 0.0));
    std::vector<double3> angularVelocities(1,make_double3(0.0, 0.0, 0.0));

    test.addCluster(positions, 
    velocities, 
    angularVelocities, 
    radius, 
    2000, 
    0);

    std::vector<double3> vertices(4);
    vertices[0] = make_double3(-1.0, -1.0, -1.0);
    vertices[1] = make_double3(1.0, -1.0, -1.0);
    vertices[2] = make_double3(1.0, 1.0, -1.0);
    vertices[3] = make_double3(-1.0, 1.0, -1.0);
    std::vector<int3> tri(2);
    tri[0] = make_int3(0, 1, 3);
    tri[1] = make_int3(1, 2, 3);

    test.addTriangleMeshWall(vertices, 
    tri, 
    make_double3(0.0, 0.0, 0.0), 
    make_double3(0.0, 0.0, 0.0), 
    make_double3(0.0, 0.0, 0.0), 
    0);

    test.setLinearContactModelForPair(0, 0, 
    1.e3, 
    0.0, 
    0.0, 
    0.0, 
    0.0, 
    0.0, 
    0.0, 
    0.0, 
    0.0, 
    0.0, 
    0.0);

    test.setTimeStep(1.e-4);
    test.setGravity(make_double3(0.0, 0.0, -9.81));
    test.setDomain(make_double3(-1.0, -1.0, -1.0), 
    make_double3(2.0, 2.0, 2.0));
    test.setNumFrames(100);

    test.solve();
}