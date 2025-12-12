#pragma once
#include "myContainer/myParticle.h"
#include "myContainer/myInteraction.h"
#include "myContainer/mySpatialGrid.h"
#include "myContainer/myWall.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

inline void computeGPUGridSizeBlockSize(size_t& gridDim, size_t& blockDim, 
    const size_t numObjects,
    const size_t maxThreadsPerBlock)
{
    if (numObjects == 0)
    {
        gridDim = 1;
        blockDim = 1;
        return;
    }
    blockDim = maxThreadsPerBlock < numObjects ? maxThreadsPerBlock : numObjects;
    gridDim = (numObjects + blockDim - 1) / blockDim;
}

__device__ __forceinline__ int3 calculateGridPosition(double3 position, const double3 minBoundary, const double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
        int((position.y - minBoundary.y) / cellSize.y),
        int((position.z - minBoundary.z) / cellSize.z));
}

__device__ __forceinline__ int calculateHash(int3 gridPosition, const int3 gridSize)
{
    return gridPosition.z * gridSize.y * gridSize.x + gridPosition.y * gridSize.x + gridPosition.x;
}

extern "C" void launchSolidParticleNeighborSearch(interactionSpringSystem& solidParticleInteractions, 
solidParticle& solidParticles, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchSolidParticleInfiniteWallNeighborSearch(interactionSpringSystem& solidParticleInfiniteWallInteractions, 
solidParticle& solidParticles, 
infiniteWall& infiniteWalls, 
objectNeighborPrefix& neighbor_s,
sortedHashValueIndex& interactionIndexRange_i,
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchSolidParticleTriangleWallNeighborSearch(interactionSpringSystem& solidParticleTriangleWallInteractions, 
solidParticle& solidParticles, 
triangleWall& triangleWalls, 
objectNeighborPrefix& neighbor_s,
sortedHashValueIndex& interactionIndexRange_t,
objectHash& triangleHash,
spatialGrid& triangleSpatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

__host__ __device__
inline bool cellCutByInfinitePlane(const double3& cellMin,
                                   const double3& cellMax,
                                   const double3& planePoint,
                                   const double3& planeNormal)
{
    double minS =  1e300;
    double maxS = -1e300;

    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                double3 corner = make_double3(
                    dx ? cellMax.x : cellMin.x,
                    dy ? cellMax.y : cellMin.y,
                    dz ? cellMax.z : cellMin.z
                );
                double s = dot(planeNormal, corner - planePoint);
                if (s < minS) minS = s;
                if (s > maxS) maxS = s;
            }
        }
    }

    return (minS <= 0.0 && maxS >= 0.0);
}

__host__ __device__
inline bool cellCutByInfiniteCylinder(const double3& cellMin,
                                      const double3& cellMax,
                                      const double3& cylPos,
                                      const double3& cylAxis,
                                      double radius)
{
    double ax = cylAxis.x;
    double ay = cylAxis.y;
    double az = cylAxis.z;
    double len2 = ax*ax + ay*ay + az*az;
    if (len2 < 1e-30) return false;
    double invLen = 1. / sqrt(len2);
    ax *= invLen;
    ay *= invLen;
    az *= invLen;

    double R2 = radius * radius;

    double minF =  1e300;
    double maxF = -1e300;

    for (int dz = 0; dz <= 1; ++dz)
    {
        double zc = dz ? cellMax.z : cellMin.z;
        for (int dy = 0; dy <= 1; ++dy)
        {
            double yc = dy ? cellMax.y : cellMin.y;
            for (int dx = 0; dx <= 1; ++dx)
            {
                double xc = dx ? cellMax.x : cellMin.x;

                double vx = xc - cylPos.x;
                double vy = yc - cylPos.y;
                double vz = zc - cylPos.z;

                double t = vx*ax + vy*ay + vz*az;

                double cx = cylPos.x + t*ax;
                double cy = cylPos.y + t*ay;
                double cz = cylPos.z + t*az;

                double dx2 = xc - cx;
                double dy2 = yc - cy;
                double dz2 = zc - cz;

                double d2 = dx2*dx2 + dy2*dy2 + dz2*dz2;
                double f  = d2 - R2;

                if (f < minF) minF = f;
                if (f > maxF) maxF = f;
            }
        }
    }

    return (minF <= 0.0) && (maxF >= 0.0);
}

__host__ __device__
inline bool pointInAABB(const double3& p,
                        const double3& bmin,
                        const double3& bmax)
{
    return (p.x >= bmin.x && p.x <= bmax.x &&
            p.y >= bmin.y && p.y <= bmax.y &&
            p.z >= bmin.z && p.z <= bmax.z);
}

__host__ __device__
inline bool segmentIntersectsTriangle(const double3& p0,
                                      const double3& p1,
                                      const double3& v0,
                                      const double3& v1,
                                      const double3& v2)
{
    const double3 dir = p1 - p0;
    const double3 e1  = v1 - v0;
    const double3 e2  = v2 - v0;

    const double3 h = cross(dir, e2);
    const double   a = dot(e1, h);

    const double eps = 1e-12;
    if (a > -eps && a < eps)
        return false;

    const double f = 1.0 / a;
    const double3 s = p0 - v0;
    const double u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    const double3 q = cross(s, e1);
    const double v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    const double t = f * dot(e2, q);
    if (t < -eps || t > 1.0 + eps)
        return false;

    return true;
}

__host__ __device__
inline bool cellCutByTriangle(const double3& cellMin,
                              const double3& cellMax,
                              const double3& v0,
                              const double3& v1,
                              const double3& v2)
{
    if (pointInAABB(v0, cellMin, cellMax) ||
        pointInAABB(v1, cellMin, cellMax) ||
        pointInAABB(v2, cellMin, cellMax))
    {
        return true;
    }

    double3 c[8];
    c[0] = make_double3(cellMin.x, cellMin.y, cellMin.z);
    c[1] = make_double3(cellMax.x, cellMin.y, cellMin.z);
    c[2] = make_double3(cellMin.x, cellMax.y, cellMin.z);
    c[3] = make_double3(cellMax.x, cellMax.y, cellMin.z);
    c[4] = make_double3(cellMin.x, cellMin.y, cellMax.z);
    c[5] = make_double3(cellMax.x, cellMin.y, cellMax.z);
    c[6] = make_double3(cellMin.x, cellMax.y, cellMax.z);
    c[7] = make_double3(cellMax.x, cellMax.y, cellMax.z);

    const int edges[12][2] = {
        {0,1}, {1,3}, {3,2}, {2,0},
        {4,5}, {5,7}, {7,6}, {6,4},
        {0,4}, {1,5}, {3,7}, {2,6}
    };

    for (int e = 0; e < 12; ++e)
    {
        const double3& a = c[edges[e][0]];
        const double3& b = c[edges[e][1]];

        if (segmentIntersectsTriangle(a, b, v0, v1, v2))
            return true;
    }

    return false;
}

__host__ __device__
inline double3 closestPointOnTriangle(const double3& point,
                                      const double3& v0,
                                      const double3& v1,
                                      const double3& v2)
{
    // Edges of the triangle
    double3 edge01 = v1 - v0;
    double3 edge02 = v2 - v0;

    // Vector from v0 to the query point
    double3 v0_to_point = point - v0;

    // -----------------------------------------------------------------
    // 1) Check if the closest point lies in the Voronoi region of v0
    // -----------------------------------------------------------------
    double dot_edge01_v0 = dot(edge01, v0_to_point);
    double dot_edge02_v0 = dot(edge02, v0_to_point);
    if (dot_edge01_v0 <= 0.0 && dot_edge02_v0 <= 0.0)
        return v0;

    // -----------------------------------------------------------------
    // 2) Check if the closest point lies in the Voronoi region of v1
    // -----------------------------------------------------------------
    double3 v1_to_point = point - v1;
    double dot_edge01_v1 = dot(edge01, v1_to_point);
    double dot_edge02_v1 = dot(edge02, v1_to_point);
    if (dot_edge01_v1 >= 0.0 && dot_edge02_v1 <= dot_edge01_v1)
        return v1;

    // -----------------------------------------------------------------
    // 3) Check if the closest point lies on edge v0-v1
    // -----------------------------------------------------------------
    double vc = dot_edge01_v0 * dot_edge02_v1 - dot_edge01_v1 * dot_edge02_v0;
    if (vc <= 0.0 && dot_edge01_v0 >= 0.0 && dot_edge01_v1 <= 0.0)
    {
        double t = dot_edge01_v0 / (dot_edge01_v0 - dot_edge01_v1); // interpolation on v0-v1
        return v0 + edge01 * t;
    }

    // -----------------------------------------------------------------
    // 4) Check if the closest point lies in the Voronoi region of v2
    // -----------------------------------------------------------------
    double3 v2_to_point = point - v2;
    double dot_edge01_v2 = dot(edge01, v2_to_point);
    double dot_edge02_v2 = dot(edge02, v2_to_point);
    if (dot_edge02_v2 >= 0.0 && dot_edge01_v2 <= dot_edge02_v2)
        return v2;

    // -----------------------------------------------------------------
    // 5) Check if the closest point lies on edge v0-v2
    // -----------------------------------------------------------------
    double vb = dot_edge01_v2 * dot_edge02_v0 - dot_edge01_v0 * dot_edge02_v2;
    if (vb <= 0.0 && dot_edge02_v0 >= 0.0 && dot_edge02_v2 <= 0.0)
    {
        double t = dot_edge02_v0 / (dot_edge02_v0 - dot_edge02_v2); // interpolation on v0-v2
        return v0 + edge02 * t;
    }

    // -----------------------------------------------------------------
    // 6) Check if the closest point lies on edge v1-v2
    // -----------------------------------------------------------------
    double va = dot_edge01_v1 * dot_edge02_v2 - dot_edge01_v2 * dot_edge02_v1;
    if (va <= 0.0 && (dot_edge02_v1 - dot_edge01_v1) >= 0.0 && (dot_edge01_v2 - dot_edge02_v2) >= 0.0)
    {
        double t = (dot_edge02_v1 - dot_edge01_v1) /
                   ((dot_edge02_v1 - dot_edge01_v1) + (dot_edge01_v2 - dot_edge02_v2)); // on v1-v2
        return v1 + (v2 - v1) * t;
    }

    // -----------------------------------------------------------------
    // 7) Closest point lies inside the triangle (use barycentric coords)
    // -----------------------------------------------------------------
    double denom = 1.0 / (va + vb + vc);
    double bary_v = vb * denom;
    double bary_w = vc * denom;

    return v0 + edge01 * bary_v + edge02 * bary_w;
}

__host__ __device__
inline bool sphereIntersectsTriangleOneSided(const double3& v0,
                                             const double3& v1,
                                             const double3& v2,
                                             const double3& normal_out,
                                             const double3& sphereCenter,
                                             double          sphereRadius)
{
    // Closest point on the triangle (in 3D) to the sphere center
    double3 closest = closestPointOnTriangle(sphereCenter, v0, v1, v2);

    // Distance from sphere center to that closest point
    double3 diff = sphereCenter - closest;
    double  dist2 = dot(diff, diff);
    double  radius2 = sphereRadius * sphereRadius;

    // No intersection if distance is larger than radius
    if (dist2 > radius2)
        return false;

    // One-sided test: require the sphere to be on the "outer" side of the triangle
    // normal_out is the outer face normal (does not strictly need to be unit length)
    if (dot(diff, normal_out) < 0.0)
        return false; // sphere is on the back side; ignore

    return true;
}