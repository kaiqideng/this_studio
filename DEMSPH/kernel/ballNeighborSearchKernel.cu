#include "ballNeighborSearchKernel.h"

__global__ void countBallInteractionsKernel(int* neighborCount,
const double3* position, 
const double* radius,
const double* inverseMass,
const int* clumpID, 
const int* hashIndex,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex[i];
                    if (idxA >= idxB) continue;
                    if (inverseMass[idxA] == 0 && inverseMass[idxB] == 0) continue;
                    if (clumpID[idxA] == clumpID[idxB]) continue;
                    double cut = 1.1 * (radA + radius[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeBallInteractionsKernel(double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed, 
int* objectPointing,
const double3* slidingSpring_old,
const double3* rollingSpring_old,
const double3* torsionSpring_old,
const int* objectPointed_old,
const int* interactionHashIndex_old,
const double3* position, 
const double* radius,
const double* inverseMass,
const int* clumpID, 
const int* hashIndex,
const int* neighborPrefixSum, 
const int* interactionStart,
const int* interactionEnd,
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex[i];
                    if (idxA >= idxB) continue;
                    if (inverseMass[idxA] == 0 && inverseMass[idxB] == 0) continue;
                    if (clumpID[idxA] == clumpID[idxB]) continue;
                    double cut = 1.1 * (radA + radius[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        slidingSpring[index_w] = make_double3(0., 0., 0.);
                        rollingSpring[index_w] = make_double3(0., 0., 0.);
                        torsionSpring[index_w] = make_double3(0., 0., 0.);
                        if (interactionStart[idxB] != -1)
                        {
                            for (int j = interactionStart[idxB]; j < interactionEnd[idxB]; j++)
                            {
                                int j1 = interactionHashIndex_old[j];
                                int idxA1 = objectPointed_old[j1];
                                if (idxA == idxA1)
                                {
                                    slidingSpring[index_w] = slidingSpring_old[j1];
                                    rollingSpring[index_w] = rollingSpring_old[j1];
                                    torsionSpring[index_w] = torsionSpring_old[j1];
                                    break;
                                }
                            }
                        }
                        count++;
                    }
                }
            }
        }
    }
}

__global__ void countBallDummyInteractionsKernel(int* neighborCount,
const double3* position, 
const double* radius,
const double3* position_dummy, 
const double* radius_dummy,
const int* hashIndex_dummy,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex_dummy[i];
                    double cut = 1.1 * (radA + radius_dummy[idxB]);
                    double3 rAB = posA - position_dummy[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeBallDummyInteractionsKernel(double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed, 
int* objectPointing,
const double3* slidingSpring_old,
const double3* rollingSpring_old,
const double3* torsionSpring_old,
const int* objectPointed_old,
const int* interactionHashIndex_old,
const double3* position, 
const double* radius,
const int* neighborPrefixSum,
const double3* position_dummy, 
const double* radius_dummy,
const int* hashIndex_dummy,
const int* interactionStart_dummy,
const int* interactionEnd_dummy,
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex_dummy[i];
                    double cut = 1.1 * (radA + radius_dummy[idxB]);
                    double3 rAB = posA - position_dummy[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        slidingSpring[index_w] = make_double3(0., 0., 0.);
                        rollingSpring[index_w] = make_double3(0., 0., 0.);
                        torsionSpring[index_w] = make_double3(0., 0., 0.);
                        if (interactionStart_dummy[idxB] != -1)
                        {
                            for (int j = interactionStart_dummy[idxB]; j < interactionEnd_dummy[idxB]; j++)
                            {
                                int j1 = interactionHashIndex_old[j];
                                int idxA1 = objectPointed_old[j1];
                                if (idxA == idxA1)
                                {
                                    slidingSpring[index_w] = slidingSpring_old[j1];
                                    rollingSpring[index_w] = rollingSpring_old[j1];
                                    torsionSpring[index_w] = torsionSpring_old[j1];
                                    break;
                                }
                            }
                        }
                        count++;
                    }
                }
            }
        }
    }
}

__global__ void countBallTriangleInteractionsKernel(int* neighborCount,
const double3* position, 
const double* radius,
const int* index0_tri, 
const int* index1_tri,
const int* index2_tri,
const int* hashIndex_tri,
const double3* globalPosition_ver,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex_tri[i];
                    double3 pos0B = globalPosition_ver[index0_tri[idxB]];
                    double3 pos1B = globalPosition_ver[index1_tri[idxB]];
                    double3 pos2B = globalPosition_ver[index2_tri[idxB]];
                    double3 n = normalize(cross(pos1B - pos0B, pos2B - pos1B));
                    double t = dot(posA - pos0B, n);
                    double overlap_plane = 0.0;
                    if (t >= 0) overlap_plane = radA - t;
                    if (t < 0) overlap_plane = radA + t;
                    if (overlap_plane > 0) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeBallTriangleInteractionsKernel(double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed, 
int* objectPointing,
const double3* slidingSpring_old,
const double3* rollingSpring_old,
const double3* torsionSpring_old,
const int* objectPointed_old,
const int* interactionHashIndex_old,
const double3* position, 
const double* radius,
const int* neighborPrefixSum,
const int* index0_tri, 
const int* index1_tri,
const int* index2_tri,
const int* hashIndex_tri,
const int* interactionStart_tri,
const int* interactionEnd_tri,
const double3* globalPosition_ver,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex_tri[i];
                    double3 pos0B = globalPosition_ver[index0_tri[idxB]];
                    double3 pos1B = globalPosition_ver[index1_tri[idxB]];
                    double3 pos2B = globalPosition_ver[index2_tri[idxB]];
                    double3 n = normalize(cross(pos1B - pos0B, pos2B - pos1B));
                    double t = dot(posA - pos0B, n);
                    double overlap_plane = 0.0;
                    if (t >= 0) overlap_plane = radA - t;
                    if (t < 0) overlap_plane = radA + t;
                    if (overlap_plane > 0)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        slidingSpring[index_w] = make_double3(0., 0., 0.);
                        rollingSpring[index_w] = make_double3(0., 0., 0.);
                        torsionSpring[index_w] = make_double3(0., 0., 0.);
                        if (interactionStart_tri[idxB] != -1)
                        {
                            for (int j = interactionStart_tri[idxB]; j < interactionEnd_tri[idxB]; j++)
                            {
                                int j1 = interactionHashIndex_old[j];
                                int idxA1 = objectPointed_old[j1];
                                if (idxA == idxA1)
                                {
                                    slidingSpring[index_w] = slidingSpring_old[j1];
                                    rollingSpring[index_w] = rollingSpring_old[j1];
                                    torsionSpring[index_w] = torsionSpring_old[j1];
                                    break;
                                }
                            }
                        }
                        count++;
                    }
                }
            }
        }
    }
}

extern "C" void launchCountBallInteractions(double3* position, 
double* radius,
double* inverseMass,
int* clumpID,
int* hashIndex, 
int* neighborCount,
int* neighborPrefixSum,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    countBallInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    radius,
    inverseMass,
    clumpID,
    hashIndex,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numBall,
    stream_GPU);
}

extern "C" void launchWriteBallInteractions(double3* position, 
double* radius,
double* inverseMass,
int* clumpID,
int* hashIndex, 
int* neighborPrefixSum,
int* interactionStart,
int* interactionEnd,

double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed,
int* objectPointing,

double3* slidingSpring_old,
double3* rollingSpring_old,
double3* torsionSpring_old,
int* objectPointed_old,
int* interactionHashIndex_old,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    writeBallInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (slidingSpring,
    rollingSpring,
    torsionSpring,
    objectPointed,
    objectPointing,
    slidingSpring_old,
    rollingSpring_old,
    torsionSpring_old,
    objectPointed_old,
    interactionHashIndex_old,
    position,
    radius,
    inverseMass,
    clumpID,
    hashIndex,
    neighborPrefixSum,
    interactionStart,
    interactionEnd,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);
}

extern "C" void launchCountBallDummyInteractions(double3* position, 
double* radius,
int* neighborCount,
int* neighborPrefixSum,

double3* position_dummy, 
double* radius_dummy,
int* hashIndex_dummy, 

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    countBallDummyInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    radius,
    position_dummy,
    radius_dummy,
    hashIndex_dummy,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numBall,
    stream_GPU);
}

extern "C" void launchWriteBallDummyInteractions(double3* position, 
double* radius,
int* neighborPrefixSum,

double3* position_dummy, 
double* radius_dummy,
int* hashIndex_dummy, 
int* interactionStart_dummy,
int* interactionEnd_dummy, 

double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed,
int* objectPointing,

double3* slidingSpring_old,
double3* rollingSpring_old,
double3* torsionSpring_old,
int* objectPointed_old,
int* interactionHashIndex_old,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    writeBallDummyInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (slidingSpring,
    rollingSpring,
    torsionSpring,
    objectPointed,
    objectPointing,
    slidingSpring_old,
    rollingSpring_old,
    torsionSpring_old,
    objectPointed_old,
    interactionHashIndex_old,
    position,
    radius,
    neighborPrefixSum,
    position_dummy,
    radius_dummy,
    hashIndex_dummy,
    interactionStart_dummy,
    interactionEnd_dummy,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);
}

extern "C" void launchCountBallTriangleInteractions(double3* position, 
double* radius,
int* neighborCount,
int* neighborPrefixSum,

int* index0_tri, 
int* index1_tri,
int* index2_tri,
int* hashIndex_tri,

double3* globalPosition_ver,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    countBallTriangleInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    radius,
    index0_tri, 
    index1_tri,
    index2_tri,
    hashIndex_tri,
    globalPosition_ver,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numBall,
    stream_GPU);
}

extern "C" void launchWriteBallTriangleInteractions(double3* position, 
double* radius,
int* neighborPrefixSum,

int* index0_tri, 
int* index1_tri,
int* index2_tri,
int* hashIndex_tri,
int* interactionStart_tri,
int* interactionEnd_tri,

double3* globalPosition_ver,

double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed,
int* objectPointing,

double3* slidingSpring_old,
double3* rollingSpring_old,
double3* torsionSpring_old,
int* objectPointed_old,
int* interactionHashIndex_old,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    writeBallTriangleInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (slidingSpring,
    rollingSpring,
    torsionSpring,
    objectPointed,
    objectPointing,
    slidingSpring_old,
    rollingSpring_old,
    torsionSpring_old,
    objectPointed_old,
    interactionHashIndex_old,
    position,
    radius,
    neighborPrefixSum,
    index0_tri, 
    index1_tri,
    index2_tri,
    hashIndex_tri,
    interactionStart_tri,
    interactionEnd_tri,
    globalPosition_ver,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);
}