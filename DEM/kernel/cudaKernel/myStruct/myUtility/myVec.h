#pragma once

#include <cuda_runtime.h> // Includes vector_types.h and CUDA definitions
#include <cmath>

// Ensure HOST_DEVICE is correctly defined for both NVCC and standard compilers
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// Use constexpr for better compile-time optimization
HOST_DEVICE constexpr double pi()
{
    return 3.14159265358979323846;
}

// --------------------------------------------------------
// Basic Operator Optimizations
// STRATEGY: Use Pass-by-Value instead of Reference.
// double3 is small (24 bytes) and fits in GPU registers.
// References cause expensive global/local memory access on GPUs.
// --------------------------------------------------------

HOST_DEVICE inline double3 operator+(double3 a, double3 b) 
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_DEVICE inline double3 operator-(double3 a, double3 b) 
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_DEVICE inline double3 operator*(double3 a, double s) 
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

HOST_DEVICE inline double3 operator*(double s, double3 a) 
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

// Optimization: Convert expensive division to multiplication by reciprocal
HOST_DEVICE inline double3 operator/(double3 a, double s) 
{
    double inv = 1.0 / s;
    return make_double3(a.x * inv, a.y * inv, a.z * inv);
}

HOST_DEVICE inline double3& operator+=(double3& a, double3 b) 
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

HOST_DEVICE inline double3& operator-=(double3& a, double3 b) 
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

HOST_DEVICE inline double3& operator*=(double3& a, double s) 
{
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

HOST_DEVICE inline double3& operator/=(double3& a, double s) 
{
    double inv = 1.0 / s;
    a.x *= inv; a.y *= inv; a.z *= inv;
    return a;
}

HOST_DEVICE inline double3 operator-(double3 a) 
{
    return make_double3(-a.x, -a.y, -a.z);
}

// --------------------------------------------------------
// Geometric Function Optimizations
// --------------------------------------------------------

HOST_DEVICE inline double dot(double3 a, double3 b) 
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HOST_DEVICE inline double3 cross(double3 a, double3 b) 
{
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

HOST_DEVICE inline double lengthSquared(double3 v) 
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

HOST_DEVICE inline double length(double3 v) 
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Optimization: Use rsqrt (Reciprocal Square Root)
// This avoids one expensive division and one square root.
HOST_DEVICE inline double3 normalize(double3 v) 
{
    double lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    
    // Avoid division by zero
    if (lenSq < 1.0e-20) return make_double3(0.0, 0.0, 0.0);
    
#ifdef __CUDA_ARCH__
    // Use hardware intrinsic rsqrt on GPU
    double invLen = rsqrt(lenSq);
#else
    // CPU fallback
    double invLen = 1.0 / sqrt(lenSq);
#endif
    
    return make_double3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// --------------------------------------------------------
// Rodrigues' Rotation Formula Optimization
// --------------------------------------------------------

HOST_DEVICE inline double3 rotateVector(double3 v, double3 angleVector) 
{
    double angle_sq = dot(angleVector, angleVector);
    
    // Small angle approximation or zero vector check to prevent NaN
    if (angle_sq < 1.0e-20) return v; 

    double angle_radians = sqrt(angle_sq);
    
    // Calculate sine and cosine
    double s, c;
#ifdef __CUDA_ARCH__
    // Optimization: Use sincos intrinsic to compute both in fewer cycles
    sincos(angle_radians, &s, &c);
#else
    s = sin(angle_radians);
    c = cos(angle_radians);
#endif

    // Optimization: Pre-calculate the normalized axis 'k'
    // k = angleVector / angle_radians
    double inv_angle = 1.0 / angle_radians;
    double3 k = make_double3(angleVector.x * inv_angle, angleVector.y * inv_angle, angleVector.z * inv_angle);
    
    // Expand Rodrigues' formula:
    // v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
    
    double k_dot_v = k.x * v.x + k.y * v.y + k.z * v.z;
    double3 k_cross_v = make_double3(
        k.y * v.z - k.z * v.y,
        k.z * v.x - k.x * v.z,
        k.x * v.y - k.y * v.x
    );

    // Common factor for the third term
    double term3_scale = k_dot_v * (1.0 - c);
    
    // Combine terms manually to avoid temporary objects
    return make_double3(
        v.x * c + k_cross_v.x * s + k.x * term3_scale,
        v.y * c + k_cross_v.y * s + k.y * term3_scale,
        v.z * c + k_cross_v.z * s + k.z * term3_scale
    );
}