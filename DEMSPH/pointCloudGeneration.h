#include "kernel/myUtility/myVec.h"
#include <cassert>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

inline std::vector<double3> generateUniformPointCloud(const double3 minBound,
const double3 maxBound,
const double dx)
{
    std::vector<double3> pts;
    if (dx <= 0.0) return pts;

    const double eps = 1e-12;

    auto countAxis = [&](double a, double b) -> int
    {
        if (b < a) std::swap(a, b);
        const double len = b - a;
        return std::max(0, static_cast<int>(std::floor(len / dx + eps)) + 1);
    };

    const int nx = countAxis(minBound.x, maxBound.x);
    const int ny = countAxis(minBound.y, maxBound.y);
    const int nz = countAxis(minBound.z, maxBound.z);

    pts.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));

    const double x0 = std::min(minBound.x, maxBound.x);
    const double y0 = std::min(minBound.y, maxBound.y);
    const double z0 = std::min(minBound.z, maxBound.z);

    const double x1 = std::max(minBound.x, maxBound.x);
    const double y1 = std::max(minBound.y, maxBound.y);
    const double z1 = std::max(minBound.z, maxBound.z);

    for (int k = 0; k < nz; ++k)
    {
        double z = z0 + k * dx;
        if (z > z1 + eps) break;

        for (int j = 0; j < ny; ++j)
        {
            double y = y0 + j * dx;
            if (y > y1 + eps) break;

            for (int i = 0; i < nx; ++i)
            {
                double x = x0 + i * dx;
                if (x > x1 + eps) break;

                pts.push_back(make_double3(x, y, z));
            }
        }
    }

    return pts;
}

struct PeriodicYBox
{
    double y0 {0.0};     // seam- (lower)
    double Ly {0.0};     // period length
    double shearX {0.0}; // x shift when wrapping in y (hex alignment)
};

inline PeriodicYBox makeHexPeriodicYBox(const double a,
                                       const int ny,
                                       const double y0)
{
    PeriodicYBox box;
    if (a <= 0.0 || ny <= 0) return box;

    const double dy = a * std::sqrt(3.0) * 0.5;
    box.y0 = y0;
    box.Ly = ny * dy;
    box.shearX = (ny & 1) ? (0.5 * a) : 0.0;
    return box;
}

// -----------------------------------------------------------------------------
// Generate hex-packed point cloud in a 2D slab (x-y), z fixed.
// - y is periodic: seams at y=y0 and y=y0+Ly
// - if ny is odd, wrapping requires x shift by shearX = 0.5*a to match lattice
// - box in x is [x0, x0 + nx*a)
// -----------------------------------------------------------------------------
inline void generateHexPointCloudPeriodicY(std::vector<double3>& points,
                                          PeriodicYBox& pbcY,
                                          const int nx,
                                          const int ny,
                                          const double a,
                                          const double3 origin) // origin = (x0,y0,z0)
{
    points.clear();
    if (nx <= 0 || ny <= 0 || a <= 0.0) { pbcY = {}; return; }

    const double dx = a;
    const double dy = a * std::sqrt(3.0) * 0.5;

    const double x0 = origin.x;
    const double y0 = origin.y;
    const double z0 = origin.z;

    // periodic info in y
    pbcY = makeHexPeriodicYBox(a, ny, y0);

    // lattice points
    points.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny));
    for (int j = 0; j < ny; ++j)
    {
        const double y = y0 + j * dy;
        const double xShift = (j & 1) ? (0.5 * dx) : 0.0;

        for (int i = 0; i < nx; ++i)
        {
            const double x = x0 + i * dx + xShift;
            points.push_back(make_double3(x, y, z0));
        }
    }
}

// Optional: apply y-periodic wrap for an arbitrary point (useful for checks)
__host__ __device__ inline double3 wrapYHex(const double3 p, const PeriodicYBox& box)
{
    double3 q = p;

    if (box.Ly <= 0.0) return q;

    // bring into [y0, y0+Ly)
    while (q.y < box.y0)       { q.y += box.Ly; q.x += box.shearX; }
    while (q.y >= box.y0+box.Ly){ q.y -= box.Ly; q.x -= box.shearX; }

    return q;
}

struct TriangleMesh
{
    std::vector<double3> vertices;

    std::vector<int> tri0;
    std::vector<int> tri1;
    std::vector<int> tri2;

    void reserveTriangles(size_t n)
    {
        tri0.reserve(n); tri1.reserve(n); tri2.reserve(n);
    }

    void addTriangle(int i0, int i1, int i2)
    {
        tri0.push_back(i0);
        tri1.push_back(i1);
        tri2.push_back(i2);
    }

    size_t numTriangles() const { return tri0.size(); }
};

inline void buildOrthonormalBasis(const double3& axisUnit, double3& u, double3& v)
{
    // pick a vector not parallel to axis
    double3 a = axisUnit;
    double3 tmp = (std::fabs(a.z) < 0.9) ? make_double3(0,0,1) : make_double3(0,1,0);
    u = normalize(cross(tmp, a));
    v = cross(a, u); // already unit if a,u are unit and orthogonal
}

// -----------------------------------------------------------------------------
// Generate cylinder triangle mesh along arbitrary axis
//
// center   : cylinder center (midpoint)
// axis     : cylinder axis direction (doesn't need unit)
// radius   : cylinder radius
// height   : cylinder height (along axis)
// nAround  : segments around circumference (>=3)
// nHeight  : segments along height (>=1)
// caps     : whether to add top/bottom caps (triangle fan)
//
// Output layout:
// - vertices: rings (nHeight+1) * nAround (+ cap centers if caps)
// - triangles: side quads split into 2 tris (+ caps)
//
// Note: Winding order is consistent but you can flip if you need outward normals.
// -----------------------------------------------------------------------------
inline TriangleMesh makeCylinderMesh(const double3& center,
                                    const double3& axis,
                                    double radius,
                                    double height,
                                    int nAround,
                                    int nHeight,
                                    bool caps = true)
{
    TriangleMesh mesh;

    assert(nAround >= 3);
    assert(nHeight >= 1);
    assert(radius > 0.0);
    assert(height > 0.0);

    const double3 w = normalize(axis);
    double3 u, v;
    buildOrthonormalBasis(w, u, v);

    const int ringCount = nHeight + 1;
    const int ringVerts = nAround;
    const int sideVertCount = ringCount * ringVerts;

    // pre-reserve
    mesh.vertices.reserve(sideVertCount + (caps ? 2 : 0));
    mesh.reserveTriangles(
        size_t(nHeight) * size_t(nAround) * 2 + (caps ? size_t(nAround) * 2 : 0)
    );

    // -------------------------
    // vertices (rings)
    // -------------------------
    for (int iz = 0; iz < ringCount; ++iz)
    {
        double t = double(iz) / double(nHeight);          // 0..1
        double z = (t - 0.5) * height;                    // -h/2 .. +h/2
        double3 c = center + z * w;

        for (int ia = 0; ia < nAround; ++ia)
        {
            double ang = (2.0 * M_PI) * (double(ia) / double(nAround));
            double3 p = c + radius * (std::cos(ang) * u + std::sin(ang) * v);
            mesh.vertices.push_back(p);
        }
    }

    auto vid = [&](int iz, int ia) -> int
    {
        // wrap ia for periodic around
        int a = ia % nAround;
        if (a < 0) a += nAround;
        return iz * nAround + a;
    };

    // -------------------------
    // side triangles
    // -------------------------
    for (int iz = 0; iz < nHeight; ++iz)
    {
        for (int ia = 0; ia < nAround; ++ia)
        {
            int i00 = vid(iz,   ia);
            int i01 = vid(iz,   ia + 1);
            int i10 = vid(iz+1, ia);
            int i11 = vid(iz+1, ia + 1);

            // two tris for each quad
            // (i00, i10, i11) and (i00, i11, i01)
            mesh.addTriangle(i00, i10, i11);
            mesh.addTriangle(i00, i11, i01);
        }
    }

    if (!caps) return mesh;

    // -------------------------
    // caps (triangle fan)
    // -------------------------
    const int bottomCenter = (int)mesh.vertices.size();
    mesh.vertices.push_back(center + (-0.5 * height) * w);

    const int topCenter = (int)mesh.vertices.size();
    mesh.vertices.push_back(center + (0.5 * height) * w);

    const int bottomRingZ = 0;
    const int topRingZ    = nHeight;

    for (int ia = 0; ia < nAround; ++ia)
    {
        int i0 = vid(bottomRingZ, ia);
        int i1 = vid(bottomRingZ, ia + 1);
        // bottom cap (flip if你想要法向朝外)
        mesh.addTriangle(bottomCenter, i1, i0);
    }

    for (int ia = 0; ia < nAround; ++ia)
    {
        int i0 = vid(topRingZ, ia);
        int i1 = vid(topRingZ, ia + 1);
        // top cap
        mesh.addTriangle(topCenter, i0, i1);
    }

    return mesh;
}