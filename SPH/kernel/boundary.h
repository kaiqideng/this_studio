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
    // SoA (per-cell arrays)
    //   cellHashStart[h] : start index in sorted list for cell h
    //   cellHashEnd[h]   : end index in sorted list for cell h
    // ---------------------------------------------------------------------
    using Storage = SoA1D<int, int>;   // (cellHashStart, cellHashEnd)
    Storage data_;

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

private:
    // ---------------------------------------------------------------------
    // SoA management
    // ---------------------------------------------------------------------
    // Allocate per-cell buffers on device (empty buffer).
    // If initToFF=true, fill both arrays with 0xFF (int = -1) on device.
    void allocateCellHashDevice(const size_t numGrids,
                                cudaStream_t stream,
                                const bool initToFF = true)
    {
        data_.allocateDevice(numGrids, stream, /*zeroFill=*/false);

        if (initToFF && numGrids > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(data_.devicePtr<0>(), 0xFF, numGrids * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(data_.devicePtr<1>(), 0xFF, numGrids * sizeof(int), stream));
        }
    }

public:
    void set(double3 minBoundary, double3 maxBoundary, double cellSizeOneDim, cudaStream_t stream)
    {
        if (maxBoundary.x <= minBoundary.x) return;
        if (maxBoundary.y <= minBoundary.y) return;
        if (maxBoundary.z <= minBoundary.z) return;
        if (cellSizeOneDim < 1.e-20) return;

        minBoundary_ = minBoundary;
        maxBoundary_ = maxBoundary;
        double3 domainSize = maxBoundary - minBoundary;
        gridSize_.x = domainSize.x > cellSizeOneDim ? int(domainSize.x / cellSizeOneDim) : 1;
        gridSize_.y = domainSize.y > cellSizeOneDim ? int(domainSize.y / cellSizeOneDim) : 1;
        gridSize_.z = domainSize.z > cellSizeOneDim ? int(domainSize.z / cellSizeOneDim) : 1;
        cellSize_.x = domainSize.x / double(gridSize_.x);
        cellSize_.y = domainSize.y / double(gridSize_.y);
        cellSize_.z = domainSize.z / double(gridSize_.z);

        allocateCellHashDevice(gridSize_.x * gridSize_.y * gridSize_.z, stream);
    }
    // ---------------------------------------------------------------------
    // Device pointers (cell hash)
    // ---------------------------------------------------------------------
    int* cellHashStart() { return data_.devicePtr<0>(); }
    int* cellHashEnd() { return data_.devicePtr<1>(); }
    size_t numGrids() { return gridSize_.x * gridSize_.y * gridSize_.z; }
};