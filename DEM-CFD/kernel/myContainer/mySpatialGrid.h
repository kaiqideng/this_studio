#pragma once
#include "myHash.h"
#include "myUtility/myVec.h"
#include <vector_types.h>

/**
 * @brief Spatial Grid Structure for neighbor search.
 * Manages the grid domain boundaries and CUDA arrays for storing cellHashValue start/end indices.
 */
struct spatialGrid 
{
    // -------------------------------------------------------------------------
    // Configuration Members (Stored on Device or passed by value/reference)
    // -------------------------------------------------------------------------
private:
    size_t  d_size { 0 };
    int3    gridSize{ make_int3(0, 0, 0) };       // Number of cells in x, y, z directions
    double3 cellSize{ make_double3(0., 0., 0.) }; // Size of one cellHashValue (h)
    double3 minBound{ make_double3(0., 0., 0.) }; // Minimum corner of the domain
    double3 maxBound{ make_double3(1., 1., 1.) }; // Maximum corner of the domain

    // -------------------------------------------------------------------------
    // Memory Management
    // -------------------------------------------------------------------------

    /**
     * @brief Allocates device memory for cellHashValue start/end arrays.
     * @param stream CUDA stream for asynchronous operation.
     */
    void alloc(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) release();
        d_size = n;

        // Allocate arrays using Async Allocator and InitMode::NEG_ONE (-1) 
        // to indicate uninitialized/empty state for sorting indices.
        cellHashValue.alloc(n, stream);
    }

    /**
     * @brief Frees all allocated device memory.
     */
    void release()
    {
        cellHashValue.release();

        d_size = 0;
    }

public:
    // -------------------------------------------------------------------------
    // Device Pointers (Cell Start/End Indices)
    // -------------------------------------------------------------------------
    sortedHashValueIndex cellHashValue;
    // -------------------------------------------------------------------------
    // Constructors & Destructor
    // -------------------------------------------------------------------------

    spatialGrid() = default;

    // Destructor (RAII): Ensures CUDA memory is freed automatically.
    ~spatialGrid() { release(); }

    // -------------------------------------------------------------------------
    // Rule of Five: Copy & Move Management
    // -------------------------------------------------------------------------

    // 1. Delete Copying (Safety)
    spatialGrid(const spatialGrid&) = delete;
    spatialGrid& operator=(const spatialGrid&) = delete;

    // 2. Move Constructor
    spatialGrid(spatialGrid&& other) noexcept { *this = std::move(other);}

    // 3. Move Assignment Operator
    spatialGrid& operator=(spatialGrid&& other) noexcept
    {
        if (this != &other)
        {
            release(); // Clean up current resources

            // Move scalars
            d_size = std::exchange(other.d_size, 0);
            minBound = std::exchange(other.minBound, make_double3(0.0, 0.0, 0.0));
            maxBound = std::exchange(other.maxBound, make_double3(1.0, 1.0, 1.0));
            cellSize = std::exchange(other.cellSize, make_double3(0.0, 0.0, 0.0));
            gridSize = std::exchange(other.gridSize, make_int3(0, 0, 0));
            // Move raw pointers (Transfer ownership)
            cellHashValue = std::move(other.cellHashValue);
        }
        return *this;
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
    }

    /**
     * @brief Resets all cellHashValue start/end indices to -1 (0xFFFFFFFF) asynchronously.
     * This prepares the grid for the next time step's particle counting.
     */
    void resetCellStartEnd(cudaStream_t stream) const
    {
        if (d_size < 1) return;
        
        // Optimization: Use CUDA_CHECK and cudaMemsetAsync for non-blocking reset.
        // 0xFFFFFFFF corresponds to -1 in 2's complement.
        cellHashValue.reset(stream);
    }

    size_t size() const 
    {
        return d_size;
    }

    const double3 &getMinBond() const
    {
        return minBound;
    }

    const double3 &getMaxBond() const
    {
        return maxBound;
    }

    const double3 &getCellSize() const
    {
        return cellSize;
    }

    const int3 &getGridSize() const
    {
        return gridSize;
    }
};