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

extern "C" void launchSolidParticleNeighborSearch(interactionSpringSystem& solidParticleInteractions, solidParticle& solidParticles, 
spatialGrid& spatialGrids, const size_t maxThreadsPerBlock, cudaStream_t stream = 0);

extern "C" void launchSolidParticleInfiniteWallNeighborSearch(interactionSpringSystem& solidParticleInfiniteWallInteractions, 
solidParticle& solidParticles, 
infiniteWall& infiniteWalls, 
objectNeighborPrefix& neighbor_si,
sortedHashValueIndex& interactionIndexRange_si,
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