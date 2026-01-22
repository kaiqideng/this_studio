#pragma once
#include "kernel/myUtility/myVec.h"

extern "C" void launchAddConstantForceouble3(double3* force,
const double3* force_external,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchAddGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampCoeff,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);