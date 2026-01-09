#pragma once
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* hashIndex, 
int* hashValue, 
const int maxHashValue, //size of ptr "start/end"
const size_t hashListSize, 
const size_t gridDim, 
const size_t blockDim, 
cudaStream_t stream);

extern "C" void buildPrefixSum(int* prefixSum, 
int* count, 
const size_t size, 
cudaStream_t stream);