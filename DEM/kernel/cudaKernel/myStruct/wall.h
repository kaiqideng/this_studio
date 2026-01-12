#pragma once
#include "myUtility/myHostDeviceArray1D.h"
#include <map>

struct triangle
{
private:
    constantHostDeviceArray1D<int> index0_;
    constantHostDeviceArray1D<int> index1_;
    constantHostDeviceArray1D<int> index2_;
    constantHostDeviceArray1D<int> wallIndex_;

    DeviceArray1D<int> hashIndex_;
    DeviceArray1D<int> hashValue_;

public:
    triangle() = default;
    ~triangle() = default;
    triangle(const triangle&) = delete;
    triangle& operator=(const triangle&) = delete;
    triangle(triangle&&) noexcept = default;
    triangle& operator=(triangle&&) noexcept = default;

    size_t hostSize() const { return index0_.hostSize(); }
    size_t deviceSize() const { return index0_.deviceSize(); }

    void addHost(int i0, int i1, int i2, int w)
    {
        index0_.addHostData(i0);
        index1_.addHostData(i1);
        index2_.addHostData(i2);
        wallIndex_.addHostData(w);
    }

    void clearHost()
    {
        index0_.clearHostData();
        index1_.clearHostData();
        index2_.clearHostData();
        wallIndex_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        index0_.download(stream);
        index1_.download(stream);
        index2_.download(stream);
        wallIndex_.download(stream);

        const size_t n = deviceSize();
        hashIndex_.allocDeviceArray(n, stream);
        hashValue_.allocDeviceArray(n, stream);
        CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, hashValue_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, hashIndex_.deviceSize() * sizeof(int), stream));
    }

    const int* index0() { return index0_.d_ptr; }
    const int* index1() { return index1_.d_ptr; }
    const int* index2() { return index2_.d_ptr; }
    const int* wallIndex() { return wallIndex_.d_ptr; }

    int* hashIndex() { return hashIndex_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }

    const std::vector<int> index0Vector() { return index0_.getHostData(); }
    const std::vector<int> index1Vector() { return index1_.getHostData(); }
    const std::vector<int> index2Vector() { return index2_.getHostData(); }
    const std::vector<int> wallIndexVector() { return wallIndex_.getHostData(); }
};


struct edge
{
private:
    constantHostDeviceArray1D<int> index0_;
    constantHostDeviceArray1D<int> index1_;
    constantHostDeviceArray1D<int> trianglePrefixSum_;
    constantHostDeviceArray1D<int> triangleIndex_;

public:
    edge() = default;
    ~edge() = default;
    edge(const edge&) = delete;
    edge& operator=(const edge&) = delete;
    edge(edge&&) noexcept = default;
    edge& operator=(edge&&) noexcept = default;

    size_t hostSize() const { return index0_.hostSize(); }
    size_t deviceSize() const { return index0_.deviceSize(); }

    void addEdgeHost(int i0, int i1, int triPrefix)
    {
        index0_.addHostData(i0);
        index1_.addHostData(i1);
        trianglePrefixSum_.addHostData(triPrefix);
    }

    void addTriangleIndexHost(int triIdx)
    {
        triangleIndex_.addHostData(triIdx);
    }

    void clearHost()
    {
        index0_.clearHostData();
        index1_.clearHostData();
        trianglePrefixSum_.clearHostData();
        triangleIndex_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        index0_.download(stream);
        index1_.download(stream);
        trianglePrefixSum_.download(stream);
        triangleIndex_.download(stream);
    }

    int* index0() { return index0_.d_ptr; }
    int* index1() { return index1_.d_ptr; }
    int* trianglesPrefixSum() { return trianglePrefixSum_.d_ptr; }
    int* triangleIndex() { return triangleIndex_.d_ptr; }
};


struct vertex
{
private:
    constantHostDeviceArray1D<double3> localPosition_;

    constantHostDeviceArray1D<int> trianglePrefixSum_;
    constantHostDeviceArray1D<int> edgePrefixSum_;

    constantHostDeviceArray1D<int> triangleIndex_;
    constantHostDeviceArray1D<int> edgeIndex_;

public:
    vertex() = default;
    ~vertex() = default;
    vertex(const vertex&) = delete;
    vertex& operator=(const vertex&) = delete;
    vertex(vertex&&) noexcept = default;
    vertex& operator=(vertex&&) noexcept = default;

    size_t hostSize() const { return localPosition_.hostSize(); }
    size_t deviceSize() const { return localPosition_.deviceSize(); }

    void addVertexHost(const double3& p0,
    int triPrefix,
    int edgePrefix)
    {
        localPosition_.addHostData(p0);
        trianglePrefixSum_.addHostData(triPrefix);
        edgePrefixSum_.addHostData(edgePrefix);
    }

    void addTriangleIndexHost(int triIdx)
    {
        triangleIndex_.addHostData(triIdx);
    }

    void addEdgeIndexHost(int eIdx)
    {
        edgeIndex_.addHostData(eIdx);
    }

    void clearHost()
    {
        localPosition_.clearHostData();
        trianglePrefixSum_.clearHostData();
        edgePrefixSum_.clearHostData();

        triangleIndex_.clearHostData();
        edgeIndex_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        localPosition_.download(stream);
        trianglePrefixSum_.download(stream);
        edgePrefixSum_.download(stream);

        triangleIndex_.download(stream);
        edgeIndex_.download(stream);
    }

    double3* localPosition() { return localPosition_.d_ptr; }
    int* trianglesPrefixSum() { return trianglePrefixSum_.d_ptr; }
    int* edgesPrefixSum() { return edgePrefixSum_.d_ptr; }
    int* triangleIndex() { return triangleIndex_.d_ptr; }
    int* edgeIndex() { return edgeIndex_.d_ptr; }

    const std::vector<double3> localPositionVector() { return localPosition_.getHostData(); }
};

struct meshWall
{
private:
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<quaternion> orientation_;
    constantHostDeviceArray1D<int> materialID_;

    triangle triangles_;
    edge edges_;
    vertex vertices_;

    HostDeviceArray1D<double3> globalVertices_;

    struct EdgeKey 
    {
        int a, b;
        EdgeKey() = default;
        EdgeKey(int i, int j) {
            if (i < j) { a = i; b = j; }
            else { a = j; b = i; }
        }
        bool operator<(const EdgeKey& other) const {
            if (a != other.a) return a < other.a;
            return b < other.b;
        }
    };

    void buildEdgesAndVertexAdjacency(const std::vector<double3>& vertices,
    const std::vector<int3>& triIndices)
    {
        edges_.clearHost();
        vertices_.clearHost();

        const int nVerts = static_cast<int>(vertices.size());
        const int nTris  = static_cast<int>(triIndices.size());
        if (nVerts == 0 || nTris == 0) return;

        std::vector<std::vector<int>> vTriAdj(nVerts);
        for (int t = 0; t < nTris; ++t)
        {
            const int i0 = triIndices[t].x;
            const int i1 = triIndices[t].y;
            const int i2 = triIndices[t].z;

            vTriAdj[i0].push_back(t);
            vTriAdj[i1].push_back(t);
            vTriAdj[i2].push_back(t);
        }

        std::map<EdgeKey, std::vector<int>> edgeToTris;

        auto addEdgeTri = [&](int i, int j, int triIdx)
        {
            EdgeKey key(i, j);
            edgeToTris[key].push_back(triIdx);
        };

        for (int t = 0; t < nTris; ++t)
        {
            const int i0 = triIndices[t].x;
            const int i1 = triIndices[t].y;
            const int i2 = triIndices[t].z;

            addEdgeTri(i0, i1, t);
            addEdgeTri(i1, i2, t);
            addEdgeTri(i2, i0, t);
        }

        const int nEdges = static_cast<int>(edgeToTris.size());

        std::vector<std::vector<int>> vEdgeAdj(nVerts);
        int edgePrefix = 0;
        for (const auto& kv : edgeToTris)
        {
            const EdgeKey& key = kv.first;
            const std::vector<int>& tris = kv.second;

            edges_.addEdgeHost(key.a, key.b, edgePrefix + static_cast<int>(tris.size()));

            for (int triIdx : tris)
            {
                edges_.addTriangleIndexHost(triIdx);
            }

            const int eIdx = static_cast<int>(edges_.hostSize()) - 1;
            vEdgeAdj[key.a].push_back(eIdx);
            vEdgeAdj[key.b].push_back(eIdx);

            edgePrefix += static_cast<int>(tris.size());
        }

        int triPrefix = 0;
        int edgePrefixV = 0;
        for (int v = 0; v < nVerts; ++v)
        {
            int triCount  = static_cast<int>(vTriAdj[v].size());
            int edgeCount = static_cast<int>(vEdgeAdj[v].size());

            vertices_.addVertexHost(vertices[v],
                            triPrefix + triCount,
                            edgePrefixV + edgeCount);

            for (int triIdx : vTriAdj[v])
            {
                vertices_.addTriangleIndexHost(triIdx);
            }
            for (int eIdx : vEdgeAdj[v])
            {
                vertices_.addEdgeIndexHost(eIdx);
            }

            triPrefix   += triCount;
            edgePrefixV += edgeCount;
        }
    }

public:
    meshWall() = default;
    ~meshWall() = default;
    meshWall(const meshWall&) = delete;
    meshWall& operator=(const meshWall&) = delete;
    meshWall(meshWall&&) noexcept = default;
    meshWall& operator=(meshWall&&) noexcept = default;

    size_t hostSize() const { return position_.hostSize(); }
    size_t deviceSize() const { return position_.deviceSize(); }

    void addWallFromMesh(const std::vector<double3>& newVertices,
    const std::vector<int3>& newTriIndices,
    const double3& pos,
    const double3& vel,
    const double3& angVel,
    const quaternion& q,
    int matID)
    {
        const int nVerts = static_cast<int>(newVertices.size());
        const int nTris  = static_cast<int>(newTriIndices.size());
        if (nVerts == 0 || nTris == 0) return;

        const int wallId = static_cast<int>(hostSize());

        position_.addHostData(pos);
        velocity_.addHostData(vel);
        angularVelocity_.addHostData(angVel);
        orientation_.addHostData(q);
        materialID_.addHostData(matID);

        // ---- collect existing vertices (local coords) ----
        std::vector<double3> allVertices = vertices_.localPositionVector();
        std::size_t vOffset = allVertices.size();

        // append new local vertices to allVertices
        allVertices.insert(allVertices.end(), newVertices.begin(), newVertices.end());

        // optionally: store global coordinates in globalVertices_
        for (int i = 0; i < nVerts; ++i)
        {
            double3 v_global = rotateVectorByQuaternion(q, newVertices[i]) + pos;
            globalVertices_.addHostData(v_global);
        }

        // ---- collect existing triangles (global vertex indices) ----
        std::vector<int> idx0 = triangles_.index0Vector();
        std::vector<int> idx1 = triangles_.index1Vector();
        std::vector<int> idx2 = triangles_.index2Vector();

        std::vector<int3> allTriIndices;
        allTriIndices.reserve(idx0.size() + newTriIndices.size());

        for (std::size_t t = 0; t < idx0.size(); ++t)
        {
            allTriIndices.push_back(make_int3(idx0[t], idx1[t], idx2[t]));
        }

        // ---- append new triangles (with global vertex indices) ----
        for (int t = 0; t < nTris; ++t)
        {
            int3 local = newTriIndices[t];
            int3 glob;
            glob.x = static_cast<int>(vOffset) + local.x;
            glob.y = static_cast<int>(vOffset) + local.y;
            glob.z = static_cast<int>(vOffset) + local.z;

            allTriIndices.push_back(glob);

            // triangles_ always stores global vertex indices + wallId
            triangles_.addHost(glob.x, glob.y, glob.z, wallId);
        }

        // rebuild edges & vertex adjacency for the whole mesh (still in local coordinates)
        buildEdgesAndVertexAdjacency(allVertices, allTriIndices);
    }

    void clearHost()
    {
        position_.clearHostData();
        velocity_.clearHostData();
        angularVelocity_.clearHostData();
        orientation_.clearHostData();
        materialID_.clearHostData();

        triangles_.clearHost();
        edges_.clearHost();
        vertices_.clearHost();

        globalVertices_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        angularVelocity_.download(stream);
        orientation_.download(stream);
        materialID_.download(stream);

        triangles_.download(stream);
        edges_.download(stream);
        vertices_.download(stream);

        globalVertices_.download(stream);
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        angularVelocity_.upload(stream);
        orientation_.upload(stream);

        globalVertices_.upload(stream);
    }

    double3* position() { return position_.d_ptr; }
    double3* velocity() { return velocity_.d_ptr; }
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    quaternion* orientation() { return orientation_.d_ptr; }
    int* materialID() { return materialID_.d_ptr; }

    triangle& triangles() { return triangles_; }
    edge& edges() { return edges_; }
    vertex& vertices() { return vertices_; }

    double3* globalVertices() { return globalVertices_.d_ptr; }

    const std::vector<double3> positionVector()
    {
        return position_.getHostData();
    }

    const std::vector<double3> velocityVector()
    {
        return velocity_.getHostData();
    }

    const std::vector<double3> angularVelocityVector()
    {
        return angularVelocity_.getHostData();
    }

    const std::vector<quaternion> orientationVector()
    {
        return orientation_.getHostData();
    }

    const std::vector<int> materialIDVector()
    {
        return materialID_.getHostData();
    }

    const std::vector<double3> globalVerticesVector()
    {
        return globalVertices_.getHostData();
    }

    double getMaxEdgeLength()
    {
        const auto verts = vertices_.localPositionVector(); 

        const auto idx0 = triangles_.index0Vector(); 
        const auto idx1 = triangles_.index1Vector();
        const auto idx2 = triangles_.index1Vector();

        if (verts.empty() || idx0.empty()) return 0.0;

        const std::size_t nTri = idx0.size();
        const std::size_t nVert = verts.size();

        double maxLen = 0.0;

        for (std::size_t t = 0; t < nTri; ++t)
        {
            const int i0 = idx0[t];
            const int i1 = idx1[t];
            const int i2 = idx2[t];

            if (i0 < 0 || i1 < 0 || i2 < 0 ||
                i0 >= static_cast<int>(nVert) ||
                i1 >= static_cast<int>(nVert) ||
                i2 >= static_cast<int>(nVert))
            {
                continue;
            }

            const double3& p0 = verts[i0];
            const double3& p1 = verts[i1];
            const double3& p2 = verts[i2];

            const double L01 = length(p0 - p1);
            const double L12 = length(p1 - p2);
            const double L20 = length(p2 - p0);

            if (L01 > maxLen) maxLen = L01;
            if (L12 > maxLen) maxLen = L12;
            if (L20 > maxLen) maxLen = L20;
        }

        return maxLen;
    }

    double3 triangleCircumcenter(const double3& a,
    const double3& b,
    const double3& c)
    {
        // Edges from vertex a
        double3 ab = b - a;
        double3 ac = c - a;

        // Triangle normal
        double3 n  = cross(ab, ac);
        double n2  = dot(n, n);   // |n|^2

        // Degenerate triangle: fall back to centroid
        if (n2 < 1e-30)
        {
            return (a + b + c) / 3.0;
        }

        // Formula:
        // O = a + ( |ac|^2 * (n × ab) + |ab|^2 * (ac × n) ) / (2 |n|^2)
        double3 term1 = cross(n,  ab) * dot(ac, ac);
        double3 term2 = cross(ac, n ) * dot(ab, ab);
        double invDen = 1.0 / (2.0 * n2);

        return a + (term1 + term2) * invDen;
    }
};