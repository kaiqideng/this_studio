#pragma once
#include "myUtility/myHostDeviceArray.h"
#include "myUtility/myVec.h"

struct spatialGrid
{
private:
    // ---------------------------------------------------------------------
    // Geometry / grid parameters (not SoA)
    // ---------------------------------------------------------------------
    double3 minBoundary_ {0.0, 0.0, 0.0};
    double3 maxBoundary_ {0.0, 0.0, 0.0};
    double3 cellSize_ {0.0, 0.0, 0.0};

    int3 gridSize_ {0, 0, 0};

private:
    // ---------------------------------------------------------------------
    // Per-cell arrays
    //   cellHashStart[h] : start index in sorted list for cell h
    //   cellHashEnd[h]   : end index in sorted list for cell h
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> cellHashStart_;
    HostDeviceArray1D<int> cellHashEnd_;

private:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateCellHashDevice(const size_t numGrids,
    cudaStream_t stream,
    const bool initToFF = true)
    {
        cellHashStart_.allocateDevice(numGrids, stream, /*zeroFill=*/false);
        cellHashEnd_.allocateDevice(numGrids, stream, /*zeroFill=*/false);

        if (initToFF && numGrids > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(cellHashStart_.d_ptr, 0xFF, numGrids * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(cellHashEnd_.d_ptr, 0xFF, numGrids * sizeof(int), stream));
        }
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    spatialGrid() = default;
    ~spatialGrid() = default;

    spatialGrid(const spatialGrid&) = delete;
    spatialGrid& operator=(const spatialGrid&) = delete;

    spatialGrid(spatialGrid&&) noexcept = default;
    spatialGrid& operator=(spatialGrid&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Getters (host-side parameters)
    // ---------------------------------------------------------------------
    double3 minimumBoundary() const { return minBoundary_; }
    double3 maximumBoundary() const { return maxBoundary_; }
    double3 cellSize() const { return cellSize_; }
    int3 gridSize() const { return gridSize_; }

public:
    // ---------------------------------------------------------------------
    // Set / initialize grid
    // ---------------------------------------------------------------------
    void set(double3 minBoundary,
    double3 maxBoundary,
    double cellSizeOneDim,
    cudaStream_t stream)
    {
        if (maxBoundary.x <= minBoundary.x) return;
        if (maxBoundary.y <= minBoundary.y) return;
        if (maxBoundary.z <= minBoundary.z) return;
        if (cellSizeOneDim < 1.e-20) return;

        minBoundary_ = minBoundary;
        maxBoundary_ = maxBoundary;

        const double3 domainSize = maxBoundary - minBoundary;

        gridSize_.x = domainSize.x > cellSizeOneDim ? int(domainSize.x / cellSizeOneDim) : 1;
        gridSize_.y = domainSize.y > cellSizeOneDim ? int(domainSize.y / cellSizeOneDim) : 1;
        gridSize_.z = domainSize.z > cellSizeOneDim ? int(domainSize.z / cellSizeOneDim) : 1;

        cellSize_.x = domainSize.x / double(gridSize_.x);
        cellSize_.y = domainSize.y / double(gridSize_.y);
        cellSize_.z = domainSize.z / double(gridSize_.z);

        allocateCellHashDevice(size_t(gridSize_.x) * size_t(gridSize_.y) * size_t(gridSize_.z), stream);
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers (cell hash)
    // ---------------------------------------------------------------------
    int* cellHashStart() { return cellHashStart_.d_ptr; }
    int* cellHashEnd() { return cellHashEnd_.d_ptr; }

    size_t numGrids() const
    {
        return size_t(gridSize_.x) * size_t(gridSize_.y) * size_t(gridSize_.z);
    }
};