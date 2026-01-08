#pragma once
#include "myStruct/myUtility/myCUDAArrayOperation.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* index, 
int* hash, 
const int maxHashValue,
const size_t hashListSize, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);