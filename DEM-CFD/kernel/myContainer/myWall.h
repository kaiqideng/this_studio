#pragma once
#include "myUtility/myMat.h"
#include "myUtility/myHostDeviceArray1D.h"
#include "myUtility/myQua.h"
#include <map>
#include <vector>
#include <vector_functions.h>

struct infiniteWall
{
private:
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> axis_;
    HostDeviceArray1D<double>  axisAngularVelocity_;
    HostDeviceArray1D<double>  radius_;
    HostDeviceArray1D<int>     materialID_;

public:
    infiniteWall() = default;
    ~infiniteWall() = default;
    infiniteWall(const infiniteWall&) = delete;
    infiniteWall& operator=(const infiniteWall&) = delete;
    infiniteWall(infiniteWall&&) noexcept = default;
    infiniteWall& operator=(infiniteWall&&) noexcept = default;

    size_t hostSize() const
    {
        return position_.hostSize();
    }

    size_t deviceSize() const
    {
        return position_.deviceSize();
    }

    void addPlane(const double3& pos,
             const double3& vel,
             const double3& n,
             int materialID)
    {
        position_.addHostData(pos);
        velocity_.addHostData(vel);
        axis_.addHostData(n);
        axisAngularVelocity_.addHostData(0.0);
        radius_.addHostData(0.0);
        materialID_.addHostData(materialID);
    }

    void addCylinder(const double3& pos,
             const double3& vel,
             const double3& axis,
             const double& angVel,
             double r,
             int materialID)
    {
        position_.addHostData(pos);
        velocity_.addHostData(vel);
        axis_.addHostData(axis);
        axisAngularVelocity_.addHostData(angVel);
        radius_.addHostData(r);
        materialID_.addHostData(materialID);
    }

    void remove(size_t index)
    {
        position_.removeHostData(index);
        velocity_.removeHostData(index);
        axis_.removeHostData(index);
        axisAngularVelocity_.removeHostData(index);
        radius_.removeHostData(index);
        materialID_.removeHostData(index);
    }

    void clearHost()
    {
        position_.clearHostData();
        velocity_.clearHostData();
        axis_.clearHostData();
        axisAngularVelocity_.clearHostData();
        radius_.clearHostData();
        materialID_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        axis_.download(stream);
        axisAngularVelocity_.download(stream);
        radius_.download(stream);
        materialID_.download(stream);
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        axis_.upload(stream);
        axisAngularVelocity_.upload(stream);
        radius_.upload(stream);
        materialID_.upload(stream);
    }

    double3* position()
    {
        return position_.d_ptr;
    }

    double3* velocity()
    {
        return velocity_.d_ptr;
    }

    double3* axis()
    {
        return axis_.d_ptr;
    }

    double* axisAngularVelocity()
    {
        return axisAngularVelocity_.d_ptr;
    }

    double* radius()
    {
        return radius_.d_ptr;
    }

    int* materialID()
    {
        return materialID_.d_ptr;
    }

    const std::vector<double3> getPositionHost()
    {
        return position_.getHostData();
    }

    const std::vector<double3> getVelocityHost()
    {
        return velocity_.getHostData();
    }

    const std::vector<double3> getAxisHost()
    {
        return axis_.getHostData();
    }

    const std::vector<double> getAxisAngularVelocityHost()
    {
        return axisAngularVelocity_.getHostData();
    }

    const std::vector<double> getRadiusHost()
    {
        return radius_.getHostData();
    }

    const std::vector<int> getMaterialIDHost()
    {
        return materialID_.getHostData();
    }
};

struct triangle
{
private:
    HostDeviceArray1D<int> index0_;
    HostDeviceArray1D<int> index1_;
    HostDeviceArray1D<int> index2_;
    HostDeviceArray1D<int> wallIndex_;

public:
    triangle() = default;
    ~triangle() = default;
    triangle(const triangle&) = delete;
    triangle& operator=(const triangle&) = delete;
    triangle(triangle&&) noexcept = default;
    triangle& operator=(triangle&&) noexcept = default;

    size_t hostSize() const { return index0_.hostSize(); }
    size_t deviceSize() const { return index0_.deviceSize(); }

    void add(int i0, int i1, int i2, int w)
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
    }

    void upload(cudaStream_t stream)
    {
        index0_.upload(stream);
        index1_.upload(stream);
        index2_.upload(stream);
        wallIndex_.upload(stream);
    }

    int* index0() { return index0_.d_ptr; }
    int* index1() { return index1_.d_ptr; }
    int* index2() { return index2_.d_ptr; }
    int* wallIndex() { return wallIndex_.d_ptr; }

    const std::vector<int> getIndex0Host() { return index0_.getHostData(); }
    const std::vector<int> getIndex1Host() { return index1_.getHostData(); }
    const std::vector<int> getIndex2Host() { return index2_.getHostData(); }
    const std::vector<int> getWallIndexHost() { return wallIndex_.getHostData(); }
};


struct edge
{
private:
    HostDeviceArray1D<int> index0_;
    HostDeviceArray1D<int> index1_;
    HostDeviceArray1D<int> numTrianglesPrefixSum_;
    HostDeviceArray1D<int> triangleIndex_;

public:
    edge() = default;
    ~edge() = default;
    edge(const edge&) = delete;
    edge& operator=(const edge&) = delete;
    edge(edge&&) noexcept = default;
    edge& operator=(edge&&) noexcept = default;

    size_t hostSize() const { return index0_.hostSize(); }
    size_t deviceSize() const { return index0_.deviceSize(); }

    void addEdge(int i0, int i1, int triCountPrefix)
    {
        index0_.addHostData(i0);
        index1_.addHostData(i1);
        numTrianglesPrefixSum_.addHostData(triCountPrefix);
    }

    void addTriangleIndex(int triIdx)
    {
        triangleIndex_.addHostData(triIdx);
    }

    void clearHost()
    {
        index0_.clearHostData();
        index1_.clearHostData();
        numTrianglesPrefixSum_.clearHostData();
        triangleIndex_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        index0_.download(stream);
        index1_.download(stream);
        numTrianglesPrefixSum_.download(stream);
        triangleIndex_.download(stream);
    }

    void upload(cudaStream_t stream)
    {
        index0_.upload(stream);
        index1_.upload(stream);
        numTrianglesPrefixSum_.upload(stream);
        triangleIndex_.upload(stream);
    }

    int* index0() { return index0_.d_ptr; }
    int* index1() { return index1_.d_ptr; }
    int* numTrianglesPrefixSum() { return numTrianglesPrefixSum_.d_ptr; }
    int* triangleIndex() { return triangleIndex_.d_ptr; }

    const std::vector<int> getIndex0Host() { return index0_.getHostData(); }
    const std::vector<int> getIndex1Host() { return index1_.getHostData(); }
};


struct vertex
{
private:
    HostDeviceArray1D<double3> localPosition_;

    HostDeviceArray1D<int> numTrianglesPrefixSum_;
    HostDeviceArray1D<int> numEdgesPrefixSum_;

    HostDeviceArray1D<int> triangleIndex_;
    HostDeviceArray1D<int> edgeIndex_;

public:
    vertex() = default;
    ~vertex() = default;
    vertex(const vertex&) = delete;
    vertex& operator=(const vertex&) = delete;
    vertex(vertex&&) noexcept = default;
    vertex& operator=(vertex&&) noexcept = default;

    size_t hostSize() const { return localPosition_.hostSize(); }
    size_t deviceSize() const { return localPosition_.deviceSize(); }

    void addVertex(const double3& p0,
                   int triPrefix,
                   int edgePrefix)
    {
        localPosition_.addHostData(p0);
        numTrianglesPrefixSum_.addHostData(triPrefix);
        numEdgesPrefixSum_.addHostData(edgePrefix);
    }

    void addTriangleIndex(int triIdx)
    {
        triangleIndex_.addHostData(triIdx);
    }

    void addEdgeIndex(int eIdx)
    {
        edgeIndex_.addHostData(eIdx);
    }

    void clearHost()
    {
        localPosition_.clearHostData();
        numTrianglesPrefixSum_.clearHostData();
        numEdgesPrefixSum_.clearHostData();
        triangleIndex_.clearHostData();
        edgeIndex_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        localPosition_.download(stream);
        numTrianglesPrefixSum_.download(stream);
        numEdgesPrefixSum_.download(stream);
        triangleIndex_.download(stream);
        edgeIndex_.download(stream);
    }

    void upload(cudaStream_t stream)
    {
        localPosition_.upload(stream);

        numTrianglesPrefixSum_.upload(stream);
        numEdgesPrefixSum_.upload(stream);
        triangleIndex_.upload(stream);
        edgeIndex_.upload(stream);
    }

    double3* localPosition() { return localPosition_.d_ptr; }
    int* numTrianglesPrefixSum() { return numTrianglesPrefixSum_.d_ptr; }
    int* numEdgesPrefixSum() { return numEdgesPrefixSum_.d_ptr; }
    int* triangleIndex() { return triangleIndex_.d_ptr; }
    int* edgeIndex() { return edgeIndex_.d_ptr; }

    const std::vector<double3> getLocalPosition() { return localPosition_.getHostData(); }
};


struct triangleWall
{
private:
    HostDeviceArray1D<double3>    position_;
    HostDeviceArray1D<double3>    velocity_;
    HostDeviceArray1D<double3>    angularVelocity_;
    HostDeviceArray1D<quaternion> orientation_;
    HostDeviceArray1D<int>        materialID_;

    triangle triangles_;
    edge     edges_;
    vertex vertices_;
    HostDeviceArray1D<double3> globalVertices_;

    struct EdgeKey {
        int a, b;
        EdgeKey() = default;
        EdgeKey(int i, int j) {
            if (i < j) { a = i; b = j; }
            else       { a = j; b = i; }
        }
        bool operator<(const EdgeKey& other) const {
            if (a != other.a) return a < other.a;
            return b < other.b;
        }
    };

    void buildEdgesAndVertexAdjacency(const std::vector<double3>& vertex,
                                    const std::vector<int3>&    triIndices)
    {
        edges_.clearHost();
        vertices_.clearHost();

        const int nVerts = static_cast<int>(vertex.size());
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

            edges_.addEdge(key.a, key.b, edgePrefix + static_cast<int>(tris.size()));

            for (int triIdx : tris)
            {
                edges_.addTriangleIndex(triIdx);
            }

            const int eIdx = static_cast<int>(edges_.hostSize()) - 1;
            vEdgeAdj[key.a].push_back(eIdx);
            vEdgeAdj[key.b].push_back(eIdx);

            edgePrefix += static_cast<int>(tris.size());
        }

        vertices_.clearHost();

        int triPrefix = 0;
        int edgePrefixV = 0;
        for (int v = 0; v < nVerts; ++v)
        {
            int triCount  = static_cast<int>(vTriAdj[v].size());
            int edgeCount = static_cast<int>(vEdgeAdj[v].size());

            vertices_.addVertex(vertex[v],
                            triPrefix + triCount,
                            edgePrefixV + edgeCount);

            for (int triIdx : vTriAdj[v])
            {
                vertices_.addTriangleIndex(triIdx);
            }
            for (int eIdx : vEdgeAdj[v])
            {
                vertices_.addEdgeIndex(eIdx);
            }

            triPrefix   += triCount;
            edgePrefixV += edgeCount;
        }
    }

public:
    triangleWall() = default;
    ~triangleWall() = default;
    triangleWall(const triangleWall&) = delete;
    triangleWall& operator=(const triangleWall&) = delete;
    triangleWall(triangleWall&&) noexcept = default;
    triangleWall& operator=(triangleWall&&) noexcept = default;

    size_t hostSize() const { return position_.hostSize(); }
    size_t deviceSize() const { return position_.deviceSize(); }

    void addWallFromMesh(const std::vector<double3>& newVertices,
                        const std::vector<int3>&    newTriIndices,
                        const double3&              pos,
                        const double3&              vel,
                        const double3&              angVel,
                        const quaternion&           q,
                        int                         matID)
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
        std::vector<double3> allVertices = vertices_.getLocalPosition();
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
        std::vector<int> idx0 = triangles_.getIndex0Host();
        std::vector<int> idx1 = triangles_.getIndex1Host();
        std::vector<int> idx2 = triangles_.getIndex2Host();

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
            triangles_.add(glob.x, glob.y, glob.z, wallId);
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
        materialID_.upload(stream);

        triangles_.upload(stream);
        edges_.upload(stream);
        vertices_.upload(stream);

        globalVertices_.upload(stream);
    }

    double3* position()        { return position_.d_ptr; }
    double3* velocity()        { return velocity_.d_ptr; }
    double3* angularVelocity() { return angularVelocity_.d_ptr; }
    quaternion* orientation()  { return orientation_.d_ptr; }
    int* materialID()          { return materialID_.d_ptr; }

    triangle& triangles() { return triangles_; }
    edge&     edgesRef()  { return edges_; }
    vertex& verticesRef()  { return vertices_; }

    double3* globalVertices() { return globalVertices_.d_ptr; }

    const std::vector<double3> getPositionHost()
    {
        return position_.getHostData();
    }

    const std::vector<double3> getVelocityHost()
    {
        return velocity_.getHostData();
    }

    const std::vector<double3> getAngularVelocityHost()
    {
        return angularVelocity_.getHostData();
    }

    const std::vector<quaternion> getOrientationHost()
    {
        return orientation_.getHostData();
    }

    const std::vector<double3> getGlobalVerticesHost()
    {
        return globalVertices_.getHostData();
    }

    double getMaxEdgeLength()
    {
        const auto verts = vertices_.getLocalPosition(); 

        const auto idx0 = triangles_.getIndex0Host(); 
        const auto idx1 = triangles_.getIndex1Host();
        const auto idx2 = triangles_.getIndex2Host();

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
};