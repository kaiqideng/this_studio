#pragma once
#include "ballIntegration.h"
#include "wallHandler.h"

__host__ __device__
inline bool isSphereEdgeContact(const double3& edgeP0,
                                const double3& edgeP1,
                                const double3& sphereCenter,
                                double         sphereRadius)
{
    // Edge direction
    double3 edge = edgeP1 - edgeP0;
    double  edgeLen2 = dot(edge, edge);
    if (edgeLen2 <= 1e-20) {
        // Degenerate edge -> treat as no edge contact
        return false;
    }

    // Project sphere center onto the infinite line of the edge
    double3 v    = sphereCenter - edgeP0;
    double  t    = dot(v, edge) / edgeLen2;

    // If projection lies outside [0,1], closest point is a vertex => not edge contact
    if (t <= 0.0 || t >= 1.0) {
        return false;
    }

    // Closest point on the segment
    double3 closest = edgeP0 + edge * t;

    // Check distance to the sphere
    double3 diff  = sphereCenter - closest;
    double  dist2 = dot(diff, diff);
    double  r2    = sphereRadius * sphereRadius;

    return dist2 <= r2;
}

enum class SphereTriangleContactType {
    None,
    Face,
    Edge,
    Vertex
};

__host__ __device__
inline SphereTriangleContactType classifySphereTriangleContact(
    const double3& sphereCenter,
    double         sphereRadius,
    const double3& v0,
    const double3& v1,
    const double3& v2,
    double3&       closestPoint  // OUT: closest point on the triangle
)
{
    double3 edge01 = v1 - v0;
    double3 edge02 = v2 - v0;
    double3 v0_to_p = sphereCenter - v0;

    const double r2 = sphereRadius * sphereRadius;

    // 1) Voronoi region of v0  -> Vertex
    double dot01_v0 = dot(edge01, v0_to_p);
    double dot02_v0 = dot(edge02, v0_to_p);
    if (dot01_v0 <= 0.0 && dot02_v0 <= 0.0) {
        closestPoint = v0;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    // 2) Voronoi region of v1 -> Vertex
    double3 v1_to_p = sphereCenter - v1;
    double dot01_v1 = dot(edge01, v1_to_p);
    double dot02_v1 = dot(edge02, v1_to_p);
    if (dot01_v1 >= 0.0 && dot02_v1 <= dot01_v1) {
        closestPoint = v1;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    // 3) Edge v0-v1 -> Edge
    double vc = dot01_v0 * dot02_v1 - dot01_v1 * dot02_v0;
    if (vc <= 0.0 && dot01_v0 >= 0.0 && dot01_v1 <= 0.0) {
        double t = dot01_v0 / (dot01_v0 - dot01_v1);
        closestPoint = v0 + edge01 * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    // 4) Voronoi region of v2 -> Vertex
    double3 v2_to_p = sphereCenter - v2;
    double dot01_v2 = dot(edge01, v2_to_p);
    double dot02_v2 = dot(edge02, v2_to_p);
    if (dot02_v2 >= 0.0 && dot01_v2 <= dot02_v2) {
        closestPoint = v2;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    // 5) Edge v0-v2 -> Edge
    double vb = dot01_v2 * dot02_v0 - dot01_v0 * dot02_v2;
    if (vb <= 0.0 && dot02_v0 >= 0.0 && dot02_v2 <= 0.0) {
        double t = dot02_v0 / (dot02_v0 - dot02_v2);
        closestPoint = v0 + edge02 * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    // 6) Edge v1-v2 -> Edge
    double va = dot01_v1 * dot02_v2 - dot01_v2 * dot02_v1;
    if (va <= 0.0 && (dot02_v1 - dot01_v1) >= 0.0 && (dot01_v2 - dot02_v2) >= 0.0) {
        double t = (dot02_v1 - dot01_v1) /
                   ((dot02_v1 - dot01_v1) + (dot01_v2 - dot02_v2));
        closestPoint = v1 + (v2 - v1) * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    // 7) Inside face -> Face
    double denom  = 1.0 / (va + vb + vc);
    double bary_v = vb * denom;
    double bary_w = vc * denom;

    closestPoint = v0 + edge01 * bary_v + edge02 * bary_w;
    double3 diff = sphereCenter - closestPoint;
    if (dot(diff, diff) <= r2) return SphereTriangleContactType::Face;

    return SphereTriangleContactType::None;
}

extern "C" void launchBallMeshWallInteractionCalculation(solidInteraction &ballMeshWallInteractions, 
ball &balls, 
meshWall &meshWalls,
contactModelParameters &contactModelParams,
interactionMap &ballTriangleInteractionMap,
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchMeshWall1stHalfIntegration(meshWall &meshWalls, 
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchMeshWall2ndHalfIntegration(meshWall &meshWalls, 
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);