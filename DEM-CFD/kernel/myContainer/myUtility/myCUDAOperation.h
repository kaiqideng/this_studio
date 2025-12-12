#pragma once
#include "myCUDA.h"

template <typename T>
inline void device_fill(T* d_ptr, size_t n, T value, cudaStream_t stream = 0);