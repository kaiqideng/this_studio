#pragma once
#include "myUtility/myCUDA.h"

/**
 * @brief Spatial Hashing Structure.
 * Manages hash values, auxiliary arrays, and sorted indices.
 */
struct objectHash 
{
private:
    size_t d_size{ 0 }; // Tracks allocated size   

public: 
    int* value{ nullptr };
    int* aux{ nullptr };
    int* index{ nullptr };

    objectHash() = default;

    // RAII Destructor: Prevents memory leaks
    ~objectHash() { release(); }

    // Disable Copying: Prevents double-free crashes
    objectHash(const objectHash&) = delete;
    objectHash& operator=(const objectHash&) = delete;

    // Enable Move Semantics: Allows efficient transfer of ownership
    objectHash(objectHash&& other) noexcept { *this = std::move(other); }
    objectHash& operator=(objectHash&& other) noexcept
    {
        if (this != &other) 
        {
            release();
            // Steal resources
            value = std::exchange(other.value, nullptr);
            aux   = std::exchange(other.aux, nullptr);
            index = std::exchange(other.index, nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    size_t size() const {return d_size;}

    void alloc(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) release();
        d_size = n;
        // Requires updated CUDA_ALLOC macro accepting stream
        CUDA_ALLOC(value, n, InitMode::NEG_ONE, stream);
        CUDA_ALLOC(aux,   n, InitMode::NEG_ONE, stream);
        CUDA_ALLOC(index, n, InitMode::NEG_ONE, stream);
    }

    void release()
    {
        if (value) { CUDA_FREE(value); value = nullptr; }
        if (aux)   { CUDA_FREE(aux);   aux   = nullptr; }
        if (index) { CUDA_FREE(index); index = nullptr; }
        d_size = 0;
    }

    void reset(cudaStream_t stream) const
    {
        if (d_size <= 0) return;
        // Async reset to -1 (0xFFFFFFFF)
        CUDA_CHECK(cudaMemsetAsync(value, 0xFF, d_size * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(aux,   0xFF, d_size * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(index, 0xFF, d_size * sizeof(int), stream));
    }
};

/**
 * @brief Neighbor Prefix Structure.
 * Used for counting sort / grid-based neighbor search.
 */
struct objectNeighborPrefix
{
private:
    size_t d_size{ 0 }; // Tracks allocated size 

public:
    int* count{ nullptr };
    int* prefixSum{ nullptr };

    objectNeighborPrefix() = default;
    ~objectNeighborPrefix() { release(); }

    objectNeighborPrefix(const objectNeighborPrefix&) = delete;
    objectNeighborPrefix& operator=(const objectNeighborPrefix&) = delete;

    objectNeighborPrefix(objectNeighborPrefix&& other) noexcept { *this = std::move(other); }
    objectNeighborPrefix& operator=(objectNeighborPrefix&& other) noexcept
    {
        if (this != &other) 
        {
            release();
            count = std::exchange(other.count, nullptr);
            prefixSum = std::exchange(other.prefixSum, nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    size_t size() const {return d_size;}

    void alloc(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) release();
        d_size = n;
        CUDA_ALLOC(count,     n, InitMode::ZERO, stream);
        CUDA_ALLOC(prefixSum, n, InitMode::ZERO, stream);
    }

    void release()
    {
        if (count)     { CUDA_FREE(count);     count = nullptr; }
        if (prefixSum) { CUDA_FREE(prefixSum); prefixSum = nullptr; }
        d_size = 0;
    }
};

/**
 * @brief Hash Indices Range Structure.
 * Stores start and end indices for sorted hash array.
 */
struct sortedHashValueIndex 
{
private:
    size_t d_size{ 0 };

public:
    int* start{ nullptr };
    int* end{ nullptr };

    sortedHashValueIndex() = default;
    ~sortedHashValueIndex() { release(); }

    sortedHashValueIndex(const sortedHashValueIndex&) = delete;
    sortedHashValueIndex& operator=(const sortedHashValueIndex&) = delete;

    sortedHashValueIndex(sortedHashValueIndex&& other) noexcept { *this = std::move(other); }
    sortedHashValueIndex& operator=(sortedHashValueIndex&& other) noexcept
    {
        if (this != &other) 
        {
            release();
            start = std::exchange(other.start, nullptr);
            end   = std::exchange(other.end, nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    size_t size() const {return d_size;}

    void alloc(size_t n, cudaStream_t stream)
    {
        if (d_size > 0) release();
        d_size = n;
        CUDA_ALLOC(start, n, InitMode::NEG_ONE, stream);
        CUDA_ALLOC(end,   n, InitMode::NEG_ONE, stream);
    }

    void release()
    {
        if (start) { CUDA_FREE(start); start = nullptr; }
        if (end)   { CUDA_FREE(end);   end   = nullptr; }
        d_size = 0;
    }

    void reset(cudaStream_t stream) const
    {
        if (d_size <= 0) return;
        CUDA_CHECK(cudaMemsetAsync(start, 0xFF, d_size * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(end,   0xFF, d_size * sizeof(int), stream));
    }
};