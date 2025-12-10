#include "neighborSearch.h"

void sortKeyValuePairs(int* d_keys, int* d_values,
                       std::size_t numObjects,
                       cudaStream_t stream)
{
    auto exec = thrust::cuda::par.on(stream);
    thrust::sort_by_key(exec,
                        d_keys, d_keys + numObjects,
                        d_values);
}

inline void inclusiveScan(int* prefixSum,
                          int* count,
                          std::size_t num,
                          cudaStream_t stream)
{
    if (num < 1) return;

    auto exec = thrust::cuda::par.on(stream);

    thrust::inclusive_scan(exec,
        thrust::device_pointer_cast(count),
        thrust::device_pointer_cast(count + num),
        thrust::device_pointer_cast(prefixSum));
}

__global__ void setInitialIndices(int* initialIndices, 
const size_t numObjects)
{
    size_t indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    initialIndices[indices] = indices;
}

__global__ void setHashAux(int* hashAux, 
int* hash, 
const size_t numObjects)
{
    size_t indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    if (indices == 0) hashAux[0] = hash[numObjects - 1];
    if (indices > 0)  hashAux[indices] = hash[indices - 1];
}

__global__ void findStartAndEnd(int* start, 
int* end, 
int* hash, 
int* hashAux, 
const size_t numObjects)
{
    size_t indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    if (indices == 0 || hash[indices] != hashAux[indices])
    {
        start[hash[indices]] = indices;
        end[hashAux[indices]] = indices;
    }
    if (indices == numObjects - 1) end[hash[indices]] = numObjects;
}

void buildHashSpans(int* start, 
int* end, 
int* sortedIndices, 
int* hash, 
int* hashAux, 
const size_t numObjects, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    if (numObjects < 1) return;

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, numObjects, maxThreadsPerBlock);

    setInitialIndices <<<grid, block, 0, stream>>> (sortedIndices, numObjects);

    sortKeyValuePairs(hash, sortedIndices, numObjects, stream);

    setHashAux <<<grid, block, 0, stream>>> (hashAux, hash, numObjects);

    findStartAndEnd <<<grid, block, 0, stream>>> (start, end, hash, hashAux, numObjects);
}

__global__ void calculateParticleHash(int* hashValue, 
double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numGrids,
const size_t numHashValues)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numHashValues) return;
    double3 pos = position[idx];
    if (minBound.x <= pos.x && pos.x < maxBound.x &&
        minBound.y <= pos.y && pos.y < maxBound.y &&
        minBound.z <= pos.z && pos.z < maxBound.z)
    {
        int3 gridPosition = calculateGridPosition(pos, minBound, cellSize);
        hashValue[idx] = calculateHash(gridPosition, gridSize);
    }
    else
    {
        hashValue[idx] = numGrids - 1;
    }
}

void updateGridCellStartEnd(spatialGrid& sptialGrids, 
objectHash& particleHash, 
double3* positions, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    sptialGrids.resetCellStartEnd(stream);
    particleHash.reset(stream);

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, particleHash.size(), maxThreadsPerBlock);

    calculateParticleHash <<< grid, block, 0, stream >>> (particleHash.value, 
    positions, 
    sptialGrids.getMinBond(), 
    sptialGrids.getMaxBond(), 
    sptialGrids.getCellSize(), 
    sptialGrids.getGridSize(), 
    sptialGrids.size(), 
    particleHash.size());

    buildHashSpans(sptialGrids.cellHashValue.start, 
    sptialGrids.cellHashValue.end, 
    particleHash.index, 
    particleHash.value, 
    particleHash.aux, 
    particleHash.size(), 
    maxThreadsPerBlock, 
    stream);
}

__global__ void setSolidParticleInteractionsKernel(int* objectPointed, int* objectPointing, 
    double3* contactForce, double3* contactTorque, 
    double3* slidingSpring, double3* rollingSpring, double3* torsionSpring, 
    int* interactionHashIndex,
    int* objectPointed_history, double3* slidingSpring_history, double3* rollingSpring_history, double3* torsionSpring_history, 
    double3* solidParticlePosition, double* effectiveRadii, double* invMass, int* clumpID, 
    int* solidParticleHashIndex, int* solidParticleNeighborCount, int* solidParticleNeighborPrefixSum, 
    int* solidParticleInteractionIndexRangeStart, int* solidParticleInteractionIndexRangeEnd,
    int* cellHashValueStart, int* cellHashValueEnd, 
    const double3 minBound, const double3 cellSize, const int3 gridSize, const size_t numGrids,
    const size_t flag,
    const size_t numParticles)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numParticles) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = solidParticleNeighborPrefixSum[idxA - 1];
    double3 posA = solidParticlePosition[idxA];
    double radA = effectiveRadii[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                if (hashB < 0 || hashB >= numGrids) continue;
                int startIndex = cellHashValueStart[hashB];
                int endIndex = cellHashValueEnd[hashB];
                if (startIndex == 0xFF) continue;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = solidParticleHashIndex[i];
                    if (idxA >= idxB) continue;
                    if (clumpID[idxA] >= 0 && clumpID[idxA] == clumpID[idxB]) continue;
					if (invMass[idxA] < 1.e-20 && invMass[idxB] < 1.e-20) continue;
                    double3 posB = solidParticlePosition[idxB];
                    double radB = effectiveRadii[idxB];
                    double3 rAB = posA - posB;
                    double dis = length(rAB);
                    double overlap = radB + radA - dis;
                    if (overlap >= 0.)
                    {
                        if (flag == 0) count++;
                        else
                        {
                            int offset_w = atomicAdd(&solidParticleNeighborCount[idxA], 1);
                            int index_w = base_w + offset_w;
                            objectPointed[index_w] = idxA;
                            objectPointing[index_w] = idxB;
                            contactForce[index_w] = make_double3(0, 0, 0);
                            contactTorque[index_w] = make_double3(0, 0, 0);
							slidingSpring[index_w] = make_double3(0, 0, 0);
							rollingSpring[index_w] = make_double3(0, 0, 0);
							torsionSpring[index_w] = make_double3(0, 0, 0);
                            if (solidParticleInteractionIndexRangeStart[idxB] != 0xFFFFFFFF)
                            {
                                for (int j = solidParticleInteractionIndexRangeStart[idxB]; j < solidParticleInteractionIndexRangeEnd[idxB]; j++)
                                {
                                    int j1 = interactionHashIndex[j];
                                    if (j1 < 0) return;
                                    int idxA1 = objectPointed_history[j1];
                                    if (idxA == idxA1)
                                    {
                                        slidingSpring[index_w] = slidingSpring_history[j1];
                                        rollingSpring[index_w] = rollingSpring_history[j1];
                                        torsionSpring[index_w] = torsionSpring_history[j1];
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (flag == 0) solidParticleNeighborCount[idxA] = count;
}

__global__ void setSolidParticleInfiniteWallInteractionsKernel(int* objectPointed, int* objectPointing, 
    double3* contactForce, double3* contactTorque, 
    double3* slidingSpring, double3* rollingSpring, double3* torsionSpring, 
    int* interactionHashIndex,
    int* objectPointed_history, double3* slidingSpring_history, double3* rollingSpring_history, double3* torsionSpring_history, 
    double3* solidParticlePosition, double* effectiveRadii, double* invMass,
    int* solidParticleNeighborCount, int* solidParticleNeighborPrefixSum, 
    int* infiniteWallInteractionIndexRangeStart, int* infiniteWallInteractionIndexRangeEnd,
    double3* position_iw, double3* axis_iw, double* radius_iw,
    const size_t numWalls,
    const size_t flag,
    const size_t numParticles)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numParticles) return;
    if (invMass[idxA] < 1.e-20) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = solidParticleNeighborPrefixSum[idxA - 1];
    double3 posA = solidParticlePosition[idxA];
    double radA = effectiveRadii[idxA];
    for (size_t idxB = 0; idxB < numWalls; idxB++)
    {
        double3 posB = position_iw[idxB];
        double3 axis = axis_iw[idxB];
        double radB = radius_iw[idxB];
        double3 n = normalize(axis);
        double overlap = radA - fabs(dot(posA - posB, n));
        if(radB > 1.e-20)
        {
            double3 p = dot(posA - posB,n) * n + posB;
            overlap = radA + radB - length(posA - p) ;
            if(overlap > radA) overlap = length(posA - p) + radA - radB;
        }
        if(overlap > 0.0)
        {
            if (flag == 0) count++;
            else
            {
                int offset_w = atomicAdd(&solidParticleNeighborCount[idxA], 1);
                int index_w = base_w + offset_w;
                objectPointed[index_w] = idxA;
                objectPointing[index_w] = idxB;
                contactForce[index_w] = make_double3(0, 0, 0);
                contactTorque[index_w] = make_double3(0, 0, 0);
				slidingSpring[index_w] = make_double3(0, 0, 0);
				rollingSpring[index_w] = make_double3(0, 0, 0);
				torsionSpring[index_w] = make_double3(0, 0, 0);
                if (infiniteWallInteractionIndexRangeStart[idxB] != 0xFFFFFFFF)
                {
                    for (int j = infiniteWallInteractionIndexRangeStart[idxB]; j < infiniteWallInteractionIndexRangeEnd[idxB]; j++)
                    {
                        int j1 = interactionHashIndex[j];
                        if (j1 < 0) return;
                        int idxA1 = objectPointed_history[j1];
                        if (idxA == idxA1)
                        {
                            slidingSpring[index_w] = slidingSpring_history[j1];
                            rollingSpring[index_w] = rollingSpring_history[j1];
                            torsionSpring[index_w] = torsionSpring_history[j1];
                            break;
                        }
                    }
                }
            }
        }
    }

    if(flag == 0) solidParticleNeighborCount[idxA] = count;
}

extern "C" void launchSolidParticleNeighborSearch(interactionSpringSystem& solidParticleInteractions, 
solidParticle& solidParticles, 
spatialGrid& spatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    updateGridCellStartEnd(spatialGrids, 
    solidParticles.hash, 
    solidParticles.position(), 
    maxThreadsPerBlock, stream);

    //debug_dump_device_array(solidParticles.hash.value, solidParticles.deviceSize(), "solidParticles.hash.value");
    //debug_dump_device_array(solidParticles.hash.index, solidParticles.deviceSize(), "solidParticles.hash.index");
    //debug_dump_device_array(spatialGrids.cellHashValue.start, spatialGrids.size(), "spatialGrids.cellHashValue.start");
    //debug_dump_device_array(spatialGrids.cellHashValue.end, spatialGrids.size(), "spatialGrids.cellHashValue.end");

    if (solidParticles.deviceSize() > 0)
    {
        size_t grid = 1, block = 1;
        computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);

        solidParticleInteractions.recordCurrentInteractionSpring(stream);

        for (size_t flag = 0; flag < 2; flag++)
        {
            //debug_dump_device_array(solidParticles.neighbor.count, solidParticles.deviceSize(), "solidParticles.neighbor.count");
            //debug_dump_device_array(solidParticles.neighbor.prefixSum, solidParticles.deviceSize(), "solidParticles.neighbor.prefixSum");

            setSolidParticleInteractionsKernel <<<grid, block, 0, stream>>> (
                solidParticleInteractions.current.objectPointed(),
                solidParticleInteractions.current.objectPointing(),
                solidParticleInteractions.current.force(),
                solidParticleInteractions.current.torque,
                solidParticleInteractions.current.slidingSpring,
                solidParticleInteractions.current.rollingSpring,
                solidParticleInteractions.current.torsionSpring,
                solidParticleInteractions.current.hash().index,
                solidParticleInteractions.history.objectPointed(),
                solidParticleInteractions.history.slidingSpring,
                solidParticleInteractions.history.rollingSpring,
                solidParticleInteractions.history.torsionSpring,
                solidParticles.position(),
                solidParticles.effectiveRadii(),
                solidParticles.inverseMass,
                solidParticles.clumpID,
                solidParticles.hash.index,
                solidParticles.neighbor.count,
                solidParticles.neighbor.prefixSum,
                solidParticles.interactionIndexRange.start,
                solidParticles.interactionIndexRange.end,
                spatialGrids.cellHashValue.start,
                spatialGrids.cellHashValue.end,
                spatialGrids.getMinBond(),
                spatialGrids.getCellSize(),
                spatialGrids.getGridSize(),
                spatialGrids.size(),
                flag,
                solidParticles.deviceSize());


            if (flag == 0)
            {
                int activeNumber = 0;
                inclusiveScan(solidParticles.neighbor.prefixSum, solidParticles.neighbor.count, solidParticles.neighbor.size(), stream);
                cuda_copy_sync(&activeNumber, solidParticles.neighbor.prefixSum + solidParticles.neighbor.size() - 1, 1, CopyDir::D2H);
                solidParticleInteractions.setCurrentActiveNumber(size_t(activeNumber), stream);
            }
        }

        solidParticles.interactionIndexRange.reset(stream);
        solidParticleInteractions.setHashValue(stream);
        buildHashSpans(solidParticles.interactionIndexRange.start, 
        solidParticles.interactionIndexRange.end, 
        solidParticleInteractions.current.hash().index, 
        solidParticleInteractions.current.hash().value, 
        solidParticleInteractions.current.hash().aux, 
        solidParticleInteractions.getActiveNumber(), maxThreadsPerBlock, stream);
    }
}

extern "C" void launchSolidParticleInfiniteWallNeighborSearch(interactionSpringSystem& solidParticleInfiniteWallInteractions, 
solidParticle& solidParticles, 
infiniteWall& infiniteWalls, 
objectNeighborPrefix& neighbor_si,
sortedHashValueIndex& interactionIndexRange_si,
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    if (infiniteWalls.deviceSize() > 0)
    {
        size_t grid = 1, block = 1;
        computeGPUGridSizeBlockSize(grid, block, solidParticles.deviceSize(), maxThreadsPerBlock);

        solidParticleInfiniteWallInteractions.recordCurrentInteractionSpring(stream);

        for (size_t flag = 0; flag < 2; flag++)
        {
            //debug_dump_device_array(solidParticles.neighbor.count, solidParticles.deviceSize(), "solidParticles.neighbor.count");
            //debug_dump_device_array(solidParticles.neighbor.prefixSum, solidParticles.deviceSize(), "solidParticles.neighbor.prefixSum");

            setSolidParticleInfiniteWallInteractionsKernel <<<grid, block, 0, stream>>> (
                solidParticleInfiniteWallInteractions.current.objectPointed(),
                solidParticleInfiniteWallInteractions.current.objectPointing(),
                solidParticleInfiniteWallInteractions.current.force(),
                solidParticleInfiniteWallInteractions.current.torque,
                solidParticleInfiniteWallInteractions.current.slidingSpring,
                solidParticleInfiniteWallInteractions.current.rollingSpring,
                solidParticleInfiniteWallInteractions.current.torsionSpring,
                solidParticleInfiniteWallInteractions.current.hash().index,
                solidParticleInfiniteWallInteractions.history.objectPointed(),
                solidParticleInfiniteWallInteractions.history.slidingSpring,
                solidParticleInfiniteWallInteractions.history.rollingSpring,
                solidParticleInfiniteWallInteractions.history.torsionSpring,
                solidParticles.position(),
                solidParticles.effectiveRadii(),
                solidParticles.inverseMass,
                neighbor_si.count,
                neighbor_si.prefixSum,
                interactionIndexRange_si.start,
                interactionIndexRange_si.end,
                infiniteWalls.position(),
                infiniteWalls.axis(),
                infiniteWalls.radius(),
                infiniteWalls.deviceSize(),
                flag,
                solidParticles.deviceSize());

            if (flag == 0)
            {
                int activeNumber = 0;
                inclusiveScan(neighbor_si.prefixSum, neighbor_si.count, neighbor_si.size(), stream);
                cuda_copy_sync(&activeNumber, neighbor_si.prefixSum + neighbor_si.size() - 1, 1, CopyDir::D2H);
                solidParticleInfiniteWallInteractions.setCurrentActiveNumber(size_t(activeNumber), stream);
            }
        }

        interactionIndexRange_si.reset(stream);
        solidParticleInfiniteWallInteractions.setHashValue(stream);
        buildHashSpans(interactionIndexRange_si.start, 
        interactionIndexRange_si.end, 
        solidParticleInfiniteWallInteractions.current.hash().index, 
        solidParticleInfiniteWallInteractions.current.hash().value, 
        solidParticleInfiniteWallInteractions.current.hash().aux, 
        solidParticleInfiniteWallInteractions.getActiveNumber(), maxThreadsPerBlock, stream);
    }
}
