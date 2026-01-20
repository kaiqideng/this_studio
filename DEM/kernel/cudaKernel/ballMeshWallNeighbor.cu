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
const size_t numTri)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri) return;
    double3 p = (vertexGlobalPosition[index0[idx]] + vertexGlobalPosition[index1[idx]] + vertexGlobalPosition[index2[idx]]) / 3.0;
    if (p.x < minBound.x) p.x = minBound.x;
    else if (p.x >= maxBound.x) p.x = maxBound.x - 0.5 * cellSize.x;
    if (p.y < minBound.y) p.y = minBound.y;
    else if (p.y >= maxBound.y) p.y = maxBound.y - 0.5 * cellSize.y;
    if (p.z < minBound.z) p.z = minBound.z;
    else if (p.z >= maxBound.z) p.z = maxBound.z - 0.5 * cellSize.z;
    int3 gridPosition = calculateGridPosition(p, minBound, cellSize);
    hashValue[idx] = calculateHash(gridPosition, gridSize);
}

extern "C" void updateTriGridCellStartEnd(spatialGrid& spatialGrids, 
int* hashIndex, 
int* hashValue, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition, 
const size_t numTri,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    calculateTriangleHash <<< gridD, blockD, 0, stream >>> (hashValue, 
    index0, 
    index1, 
    index2, 
    vertexGlobalPosition, 
    spatialGrids.minBound, 
    spatialGrids.maxBound, 
    spatialGrids.cellSize, 
    spatialGrids.gridSize,
    numTri);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(spatialGrids.cellHashStart(), 0xFF, spatialGrids.deviceSize() * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(spatialGrids.cellHashEnd(), 0xFF, spatialGrids.deviceSize() * sizeof(int), stream));
    buildHashStartEnd(spatialGrids.cellHashStart(), 
    spatialGrids.cellHashEnd(),
    hashIndex,
    hashValue,
    spatialGrids.deviceSize(),
    numTri,
    gridD, 
    blockD, 
    stream);
}

__global__ void countBallTriangleInteractionsKernel(int* interactionMapCountA, 
double3* ballPosition, 
const double* radius, 
const double* invMass, 
int* triangleHashIndex, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition,
int* cellHashStart, 
int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;
    if(invMass[idxA] == 0) return;
    interactionMapCountA[idxA] = 0;

    int count = 0;
    double3 posA = ballPosition[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) {gridPositionA.x = 0; gridStart.x = 0;}
    if (gridPositionA.x >= gridSize.x - 1) {gridPositionA.x = gridSize.x - 1; gridEnd.x = 0;}
    if (gridPositionA.y <= 0) {gridPositionA.y = 0; gridStart.y = 0;}
    if (gridPositionA.y >= gridSize.y - 1) {gridPositionA.y = gridSize.y - 1; gridEnd.y = 0;}
    if (gridPositionA.z <= 0) {gridPositionA.z = 0; gridStart.z = 0;}
    if (gridPositionA.z >= gridSize.z - 1) {gridPositionA.z = gridSize.z - 1; gridEnd.z = 0;}
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = triangleHashIndex[i];
                    double3 pos0B = vertexGlobalPosition[index0[idxB]];
                    double3 pos1B = vertexGlobalPosition[index1[idxB]];
                    double3 pos2B = vertexGlobalPosition[index2[idxB]];
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
    interactionMapCountA[idxA] = count;
}

__global__ void writeBallTriangleInteractionsKernel(int* objectPointed, 
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
int* interactionMapPrefixSumA, 
int* interactionMapStartB, 
int* interactionMapEndB,
int* cellHashStart, 
int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize,
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;
    if(invMass[idxA] < 1.e-20) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = interactionMapPrefixSumA[idxA - 1];
    double3 posA = ballPosition[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) {gridPositionA.x = 0; gridStart.x = 0;}
    if (gridPositionA.x >= gridSize.x - 1) {gridPositionA.x = gridSize.x - 1; gridEnd.x = 0;}
    if (gridPositionA.y <= 0) {gridPositionA.y = 0; gridStart.y = 0;}
    if (gridPositionA.y >= gridSize.y - 1) {gridPositionA.y = gridSize.y - 1; gridEnd.y = 0;}
    if (gridPositionA.z <= 0) {gridPositionA.z = 0; gridStart.z = 0;}
    if (gridPositionA.z >= gridSize.z - 1) {gridPositionA.z = gridSize.z - 1; gridEnd.z = 0;}
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellHashEnd[hashB];
                int countInOneCell = 0;
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
                        int index_w = base_w + count;
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
                        count++;
                    }
                }
            }
        }
    }
}

extern "C" void launchBallTriangleNeighborSearch(solidInteraction& ballTriangleInteractions, 
interactionMap &ballTriangleInteractionMap,
ball& balls, 
meshWall& meshWalls,
spatialGrid& triangleSpatialGrids,
const size_t maxThreadsPerBlock,
cudaStream_t stream)
{
    int activeNumber = 0;
    size_t gridD = 1, blockD = 1;
    if (setGPUGridBlockDim(gridD, blockD, meshWalls.triangles().deviceSize(), maxThreadsPerBlock))
    {
        updateTriGridCellStartEnd(triangleSpatialGrids,
        meshWalls.triangles().hashIndex(),
        meshWalls.triangles().hashValue(),
        meshWalls.triangles().index0(),
        meshWalls.triangles().index1(),
        meshWalls.triangles().index2(),
        meshWalls.globalVertices(),
        meshWalls.triangles().deviceSize(),
        gridD,
        blockD,
        stream);
    }

    ballTriangleInteractions.updateHistory(stream);

    if (setGPUGridBlockDim(gridD, blockD, balls.deviceSize(), maxThreadsPerBlock))
    {
        countBallTriangleInteractionsKernel <<<gridD, blockD, 0, stream>>> (ballTriangleInteractionMap.countA(),
        balls.position(),
        balls.radius(),
        balls.inverseMass(),
        meshWalls.triangles().hashIndex(),
        meshWalls.triangles().index0(),
        meshWalls.triangles().index1(),
        meshWalls.triangles().index2(),
        meshWalls.globalVertices(),
        triangleSpatialGrids.cellHashStart(),
        triangleSpatialGrids.cellHashEnd(),
        triangleSpatialGrids.minBound,
        triangleSpatialGrids.cellSize,
        triangleSpatialGrids.gridSize,
        balls.deviceSize());

        //debug_dump_device_array(ballTriangleInteractionMap.countA(), ballTriangleInteractionMap.ASize(), "ballInteractionMap.countA");
        buildPrefixSum(ballTriangleInteractionMap.prefixSumA(), 
        ballTriangleInteractionMap.countA(), 
        ballTriangleInteractionMap.ASize(), stream);
        cuda_copy_sync(&activeNumber, ballTriangleInteractionMap.prefixSumA() + ballTriangleInteractionMap.ASize() - 1, 1, CopyDir::D2H);
        ballTriangleInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);

        writeBallTriangleInteractionsKernel <<<gridD, blockD, 0, stream>>> (ballTriangleInteractions.objectPointed(),
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
        ballTriangleInteractionMap.prefixSumA(),
        ballTriangleInteractionMap.startB(),
        ballTriangleInteractionMap.endB(),
        triangleSpatialGrids.cellHashStart(),
        triangleSpatialGrids.cellHashEnd(),
        triangleSpatialGrids.minBound,
        triangleSpatialGrids.cellSize,
        triangleSpatialGrids.gridSize,
        balls.deviceSize());

        CUDA_CHECK(cudaMemsetAsync(ballTriangleInteractionMap.startB(), 0xFF, ballTriangleInteractionMap.BSize() * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(ballTriangleInteractionMap.endB(), 0xFF, ballTriangleInteractionMap.BSize() * sizeof(int), stream));
        ballTriangleInteractionMap.resizeHashIndex(static_cast<size_t>(activeNumber), stream);
        if (setGPUGridBlockDim(gridD, blockD, static_cast<size_t>(activeNumber), maxThreadsPerBlock))
        {
            buildHashStartEnd(ballTriangleInteractionMap.startB(), 
            ballTriangleInteractionMap.endB(), 
            ballTriangleInteractionMap.hashIndex(), 
            ballTriangleInteractions.objectPointing(),
            ballTriangleInteractionMap.BSize(),
            static_cast<size_t>(activeNumber), 
            gridD,
            blockD,
            stream);
        }
    }
}