#pragma once
#include "myUtility/myHostDeviceArray1D.h"

struct spatialGrid
{
public:
    int3    gridSize{ make_int3(0, 0, 0) };       // Number of cells in x, y, z directions
    double3 cellSize{ make_double3(0., 0., 0.) }; // Size of one cellHashValue (h)
    double3 minBound{ make_double3(0., 0., 0.) }; // Minimum corner of the domain
    double3 maxBound{ make_double3(0., 0., 0.) }; // Maximum corner of the domain

private:
    DeviceArray1D<int> cellHashStart_;
    DeviceArray1D<int> cellHashEnd_;

public:
    spatialGrid() = default;
    ~spatialGrid() = default;
    spatialGrid(const spatialGrid&) = delete;
    spatialGrid& operator=(const spatialGrid&) = delete;
    spatialGrid(spatialGrid&&) noexcept = default;
    spatialGrid& operator=(spatialGrid&&) noexcept = default;

    size_t deviceSize() const { return cellHashStart_.deviceSize(); }

    void alloc(size_t n, cudaStream_t stream)
    {
        cellHashStart_.allocDeviceArray(n, stream);
        cellHashEnd_.allocDeviceArray(n, stream);
    }

    void set(double3 domainOrigin, double3 domainSize, double cellSizeOneDim, cudaStream_t stream)
    {
        if(cellSizeOneDim < 1.e-20) return;

        minBound = domainOrigin;
        maxBound = domainOrigin + domainSize;
        gridSize.x = domainSize.x > cellSizeOneDim ? int(domainSize.x / cellSizeOneDim) : 1;
        gridSize.y = domainSize.y > cellSizeOneDim ? int(domainSize.y / cellSizeOneDim) : 1;
        gridSize.z = domainSize.z > cellSizeOneDim ? int(domainSize.z / cellSizeOneDim) : 1;
        cellSize.x = domainSize.x / double(gridSize.x);
        cellSize.y = domainSize.y / double(gridSize.y);
        cellSize.z = domainSize.z / double(gridSize.z);

        alloc(gridSize.x * gridSize.y * gridSize.z + 1, stream);
        cellInit(stream);
    }

    void cellInit(cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemsetAsync(cellHashStart_.d_ptr, 0xFF, cellHashStart_.deviceSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(cellHashEnd_.d_ptr,   0xFF, cellHashEnd_.deviceSize() * sizeof(int), stream));
    }

    int* cellHashStart() { return cellHashStart_.d_ptr; }
    int* cellHashEnd()   { return cellHashEnd_.d_ptr; }
};