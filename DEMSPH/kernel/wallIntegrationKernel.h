#pragma once
#include "myQua.h"

extern "C" void launchWallIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity,
quaternion* orientation, 
const double timeStep,
const size_t numWall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchUpdateWallVertexGlobalPosition(double3* globalPosition_v,
double3* localPosition_v,
int* wallIndex_v,

double3* position_w,
quaternion* orientation_w,

const size_t numVertex,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);