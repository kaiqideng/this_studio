#pragma once
#include "myUtility/myMat.h"

extern "C" void launchBall1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* radius, 
double* invMass, 
int* clumpID, 

const double3 gravity, 
const double timeStep,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchBall2ndHalfIntegration(double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* radius, 
double* invMass, 
int* clumpID,

const double3 gravity, 
const double timeStep, 

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchClump1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 
const int* pebbleStart, 
const int* pebbleEnd,

double3* position_p, 
double3* velocity_p, 
double3* angularVelocity_p, 

const double3 gravity, 
const double timeStep,

const size_t numClump,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchClump2ndHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 
const int* pebbleStart, 
const int* pebbleEnd,

double3* position_p, 
double3* velocity_p, 
double3* angularVelocity_p, 
double3* force_p, 
double3* torque_p,

const double3 gravity, 
const double timeStep,

const size_t numClump,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);
