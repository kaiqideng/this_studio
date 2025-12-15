#include "ballMeshWallNeighbor.h"

__global__ void calculateTriangleHash(int* hashValue, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition, 
const double3 minBound, 
const double3 maxBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numGrids,
const size_t numTri)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri) return;
    double3 p = (vertexGlobalPosition[index0[idx]] + vertexGlobalPosition[index1[idx]] + vertexGlobalPosition[index2[idx]]) / 3.0;
    if (minBound.x <= p.x && p.x < maxBound.x &&
        minBound.y <= p.y && p.y < maxBound.y &&
        minBound.z <= p.z && p.z < maxBound.z)
    {
        int3 gridPosition = calculateGridPosition(p, minBound, cellSize);
        hashValue[idx] = calculateHash(gridPosition, gridSize);
    }
    else
    {
        hashValue[idx] = numGrids - 1;
    }
}

extern "C" void updateTriGridCellStartEnd(spatialGrid& sptialGrids, 
int* hashIndex, 
int* hashValue, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition, 
const size_t numTri,
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    sptialGrids.cellInit(stream);

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, numTri, maxThreadsPerBlock);

    calculateTriangleHash <<< grid, block, 0, stream >>> (hashValue, 
    index0, 
    index1, 
    index2, 
    vertexGlobalPosition, 
    sptialGrids.minBound, 
    sptialGrids.maxBound, 
    sptialGrids.cellSize, 
    sptialGrids.gridSize, 
    sptialGrids.deviceSize(), 
    numTri);
    CUDA_CHECK(cudaGetLastError());

    buildHashStartEnd(sptialGrids.cellHashStart(), 
    sptialGrids.cellHashEnd(), 
    hashIndex,
    hashValue,
    static_cast<int>(sptialGrids.deviceSize()),
    numTri,
    maxThreadsPerBlock, 
    stream);
}

__global__ void setBallTriangleInteractionsKernel(int* objectPointed, 
int* objectPointing, 
double3* contactForce, 
double3* contactTorque, 
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
int* objectPointed_history, 
double3* slidingSpring_history, 
double3* rollingSpring_history, 
double3* torsionSpring_history, 
double3* ballPosition, 
const double* radius, 
const double* invMass, 
int* triangleHashIndex, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition,
int* interactionMapHashIndex,
int* interactionMapCountA, 
int* interactionMapPrefixSumA, 
int* interactionMapStartB, 
int* interactionMapEndB,
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numGrids,
const size_t flag,
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;

    interactionMapCountA[idxA] = 0;
    if(flag == 0) interactionMapPrefixSumA[idxA] = 0;

    if(invMass[idxA] == 0) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = interactionMapPrefixSumA[idxA - 1];
    double3 posA = ballPosition[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if(gridPositionB.x < 0 || gridPositionB.y < 0 ||gridPositionB.z < 0) continue;
                if(gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y ||gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                int endIndex = cellEnd[hashB];
                if (startIndex == 0xFF) continue;
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = triangleHashIndex[i];

                    double3 pos0B = vertexGlobalPosition[index0[idxB]];
                    double3 pos1B = vertexGlobalPosition[index1[idxB]];
                    double3 pos2B = vertexGlobalPosition[index2[idxB]];
                    double3 n = normalize(cross(pos1B - pos0B, pos2B - pos1B));
                    double t = dot(posA - pos0B, n);
                    double overlap_plane = 0.0;
                    if(t >= 0) overlap_plane = radA - t;
                    if(t < 0) overlap_plane = radA + t;
                    if(overlap_plane > 0)
                    {
                        if (flag == 0) count++;
                        else
                        {
                            int offset_w = atomicAdd(&interactionMapCountA[idxA], 1);
                            int index_w = base_w + offset_w;
                            objectPointed[index_w] = idxA;
                            objectPointing[index_w] = idxB;
                            contactForce[index_w] = make_double3(0, 0, 0);
                            contactTorque[index_w] = make_double3(0, 0, 0);
							slidingSpring[index_w] = make_double3(0, 0, 0);
							rollingSpring[index_w] = make_double3(0, 0, 0);
							torsionSpring[index_w] = make_double3(0, 0, 0);
                            if (interactionMapStartB[idxB] != 0xFFFFFFFF)
                            {
                                for (int j = interactionMapStartB[idxB]; j < interactionMapEndB[idxB]; j++)
                                {
                                    int j1 = interactionMapHashIndex[j];
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
    if (flag == 0) interactionMapCountA[idxA] = count;
}

extern "C" void launchBallTriangleNeighborSearch(solidInteraction& ballTriangleInteractions, 
interactionMap &ballTriangleInteractionMap,
ball& balls, 
meshWall& meshWalls,
spatialGrid& triangleSpatialGrids, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream)
{
    if(triangleSpatialGrids.deviceSize() == 0) return;
    updateTriGridCellStartEnd(triangleSpatialGrids,
    meshWalls.triangles().hashIndex(),
    meshWalls.triangles().hashValue(),
    meshWalls.triangles().index0(),
    meshWalls.triangles().index1(),
    meshWalls.triangles().index2(),
    meshWalls.globalVertices(),
    meshWalls.triangles().deviceSize(),
    maxThreadsPerBlock,
    stream);

    size_t grid = 1, block = 1;
    computeGPUGridSizeBlockSize(grid, block, balls.deviceSize(), maxThreadsPerBlock);

    for (size_t flag = 0; flag < 2; flag++)
    {
        setBallTriangleInteractionsKernel <<<grid, block, 0, stream>>> (ballTriangleInteractions.objectPointed(),
        ballTriangleInteractions.objectPointing(),
        ballTriangleInteractions.force(),
        ballTriangleInteractions.torque(),
        ballTriangleInteractions.slidingSpring(),
        ballTriangleInteractions.rollingSpring(),
        ballTriangleInteractions.torsionSpring(),
        ballTriangleInteractions.objectPointedHistory(),
        ballTriangleInteractions.slidingSpringHistory(),
        ballTriangleInteractions.rollingSpringHistory(),
        ballTriangleInteractions.torsionSpringHistory(),
        balls.position(),
        balls.radius(),
        balls.inverseMass(),
        meshWalls.triangles().hashIndex(),
        meshWalls.triangles().index0(),
        meshWalls.triangles().index1(),
        meshWalls.triangles().index2(),
        meshWalls.globalVertices(),
        ballTriangleInteractionMap.hashIndex(),
        ballTriangleInteractionMap.countA(),
        ballTriangleInteractionMap.prefixSumA(),
        ballTriangleInteractionMap.startB(),
        ballTriangleInteractionMap.endB(),
        triangleSpatialGrids.cellHashStart(),
        triangleSpatialGrids.cellHashEnd(),
        triangleSpatialGrids.minBound,
        triangleSpatialGrids.cellSize,
        triangleSpatialGrids.gridSize,
        triangleSpatialGrids.deviceSize(),
        flag,
        balls.deviceSize());

        if (flag == 0)
        {
            int activeNumber = 0;
            auto exec = thrust::cuda::par.on(stream);
            thrust::inclusive_scan(exec,
            thrust::device_pointer_cast(ballTriangleInteractionMap.countA()),
            thrust::device_pointer_cast(ballTriangleInteractionMap.countA() + ballTriangleInteractionMap.ASize()),
            thrust::device_pointer_cast(ballTriangleInteractionMap.prefixSumA()));
            cuda_copy_sync(&activeNumber, ballTriangleInteractionMap.prefixSumA() + ballTriangleInteractionMap.ASize() - 1, 1, CopyDir::D2H);
            ballTriangleInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);
        }
    }

    ballTriangleInteractionMap.hashInit(ballTriangleInteractions.objectPointing(), ballTriangleInteractions.activeSize(), stream);
    buildHashStartEnd(ballTriangleInteractionMap.startB(), 
    ballTriangleInteractionMap.endB(), 
    ballTriangleInteractionMap.hashIndex(), 
    ballTriangleInteractionMap.hashValue(),
    static_cast<int>(ballTriangleInteractionMap.BSize()),
    ballTriangleInteractionMap.activeHashSize(), 
    maxThreadsPerBlock, 
    stream);
}