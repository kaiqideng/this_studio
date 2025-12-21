#include "SPHNeighborSearch.h"

__global__ void countSPHInteractionsKernel(double3* position, 
const double* smoothLength, 
int* SPHHashIndex, 
int* neighborCountA, 
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPHs)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPHs) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = SPHHashIndex[i];
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);;
                    double3 posB = position[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCountA[idxA] = count;
}

__global__ void writeSPHInteractionsKernel(int* objectPointed, 
int* objectPointing, 
double3* position, 
const double* smoothLength, 
int* SPHHashIndex, 
int* neighborPrefixSumA, 
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPHs)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPHs) return;

    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSumA[idxA - 1];
    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                int countInOneCell = 0;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = SPHHashIndex[i];
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 posB = position[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + countInOneCell;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        countInOneCell++;
                    }
                }
            }
        }
    }
}

__global__ void countSPHVirtualInteractionsKernel(double3* position_SPH, 
const double* smoothLength, 
double3* position_virtual,
int* virtualHashIndex, 
int* neighborCountA, 
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPHs)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPHs) return;
    int count = 0;

    double3 posA = position_SPH[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = virtualHashIndex[i];
                    double cut = 2.0 * radA;
                    double3 posB = position_virtual[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCountA[idxA] = count;
}

__global__ void writeSPHVirtualInteractionsKernel(int* objectPointed, 
int* objectPointing, 
double3* force_interaction, 
double3* position_SPH, 
const double* smoothLength, 
double3* position_virtual, 
int* virtualHashIndex, 
int* neighborPrefixSumA, 
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPHs)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPHs) return;

    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSumA[idxA - 1];
    double3 posA = position_SPH[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
                int countInOneCell = 0;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = virtualHashIndex[i];
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 posB = position_virtual[idxB];
                    double3 rAB = posA - posB;
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + countInOneCell;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        force_interaction[index_w] = make_double3(0.0, 0.0, 0.0);
                        countInOneCell++;
                    }
                }
            }
        }
    }
}

extern "C" void launchSPHNeighborSearch(SPHInteraction& SPHInteractions, 
interactionMap& SPHInteractionMap,
SPH& SPHs, 
SPHInteraction& SPHVirtualInteractions, 
interactionMap& SPHVirtualInteractionMap,
virtualParticle& virtualParticles, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    size_t grid = 1, block = 1;

    //SPH-SPH
    updateGridCellStartEnd(spatialGrids,
    SPHs.hashIndex(),
    SPHs.hashValue(),
    SPHs.position(),
    SPHs.deviceSize(),
    maxThreadsPerBlock,
    stream);

    computeGPUGridSizeBlockSize(grid, block, SPHs.deviceSize(), maxThreadsPerBlock);
    countSPHInteractionsKernel <<<grid, block, 0, stream>>> (SPHs.position(),
    SPHs.smoothLength(),
    SPHs.hashIndex(),
    SPHInteractionMap.countA(),
    spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    spatialGrids.minBound,
    spatialGrids.cellSize,
    spatialGrids.gridSize,
    SPHs.deviceSize());

    int activeNumber = 0;
    auto exec = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(exec,
    thrust::device_pointer_cast(SPHInteractionMap.countA()),
    thrust::device_pointer_cast(SPHInteractionMap.countA() + SPHInteractionMap.ASize()),
    thrust::device_pointer_cast(SPHInteractionMap.prefixSumA()));
    cuda_copy_sync(&activeNumber, SPHInteractionMap.prefixSumA() + SPHInteractionMap.ASize() - 1, 1, CopyDir::D2H);
    SPHInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);

    writeSPHInteractionsKernel <<<grid, block, 0, stream>>> (SPHInteractions.objectPointed(),
    SPHInteractions.objectPointing(),
    SPHs.position(),
    SPHs.smoothLength(),
    SPHs.hashIndex(),
    SPHInteractionMap.prefixSumA(),
    spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    spatialGrids.minBound,
    spatialGrids.cellSize,
    spatialGrids.gridSize,
    SPHs.deviceSize());

    //SPH-virtual
    updateGridCellStartEnd(spatialGrids,
    virtualParticles.hashIndex(),
    virtualParticles.hashValue(),
    virtualParticles.position(),
    virtualParticles.deviceSize(),
    maxThreadsPerBlock,
    stream);

    countSPHVirtualInteractionsKernel <<<grid, block, 0, stream>>> (SPHs.position(),
    SPHs.smoothLength(),
    virtualParticles.position(),
    virtualParticles.hashIndex(),
    SPHVirtualInteractionMap.countA(),
    spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    spatialGrids.minBound,
    spatialGrids.cellSize,
    spatialGrids.gridSize,
    SPHs.deviceSize());

    int activeNumber1 = 0;
    auto exec1 = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(exec1,
    thrust::device_pointer_cast(SPHVirtualInteractionMap.countA()),
    thrust::device_pointer_cast(SPHVirtualInteractionMap.countA() + SPHVirtualInteractionMap.ASize()),
    thrust::device_pointer_cast(SPHVirtualInteractionMap.prefixSumA()));
    cuda_copy_sync(&activeNumber1, SPHVirtualInteractionMap.prefixSumA() + SPHVirtualInteractionMap.ASize() - 1, 1, CopyDir::D2H);
    SPHVirtualInteractions.setActiveSize(static_cast<size_t>(activeNumber1), stream);

    writeSPHVirtualInteractionsKernel <<<grid, block, 0, stream>>> (SPHVirtualInteractions.objectPointed(),
    SPHVirtualInteractions.objectPointing(),
    SPHVirtualInteractions.force(),
    SPHs.position(),
    SPHs.smoothLength(),
    virtualParticles.position(),
    virtualParticles.hashIndex(),
    SPHVirtualInteractionMap.prefixSumA(),
    spatialGrids.cellHashStart(),
    spatialGrids.cellHashEnd(),
    spatialGrids.minBound,
    spatialGrids.cellSize,
    spatialGrids.gridSize,
    SPHs.deviceSize());
}