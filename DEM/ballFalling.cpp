#include "kernel/DEMBallMeshWallSolver.h"

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

}