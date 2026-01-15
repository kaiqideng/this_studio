#pragma once
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* hashIndex, 
int* hashValue, 
const size_t startEndSize,
const size_t hashListSize, 
const size_t gridD, 
const size_t blockD, 
cudaStream_t stream);

extern "C" void buildPrefixSum(int* prefixSum, 
int* count, 
const size_t size, 
cudaStream_t stream);