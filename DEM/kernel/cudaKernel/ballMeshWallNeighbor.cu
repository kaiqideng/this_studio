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
const size_t gridDim,
const size_t blockDim,
cudaStream_t stream)
{
    calculateTriangleHash <<< gridDim, blockDim, 0, stream >>> (hashValue, 
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

    buildHashStartEnd(sptialGrids.cellHashStart(), 
    sptialGrids.cellHashEnd(),
    hashIndex,
    hashValue,
    static_cast<int>(sptialGrids.deviceSize()),
    numTri,
    gridDim, 
    blockDim, 
    stream);
}

__global__ void countBallTriangleInteractionsKernel(double3* ballPosition, 
const double* radius, 
const double* invMass, 
int* triangleHashIndex, 
const int* index0, 
const int* index1, 
const int* index2, 
double3* vertexGlobalPosition,
int* interactionMapCountA, 
int* cellStart, 
int* cellEnd, 
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
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                if (gridPositionB.x < 0 || gridPositionB.y < 0 || gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y || gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
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
int* cellStart, 
int* cellEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize,
const size_t numBalls)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBalls) return;
    if(invMass[idxA] < 1.e-20) return;

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
                if (gridPositionB.x < 0 || gridPositionB.y < 0 || gridPositionB.z < 0) continue;
                if (gridPositionB.x >= gridSize.x || gridPositionB.y >= gridSize.y || gridPositionB.z >= gridSize.z) continue;
                int hashB = calculateHash(gridPositionB, gridSize);
                int startIndex = cellStart[hashB];
                if (startIndex == 0xFF) continue;
                int endIndex = cellEnd[hashB];
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
                        int index_w = base_w + countInOneCell;
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
                        countInOneCell++;
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
    size_t gridDim = 1, blockDim = 1;
    if (setGPUGridBlockDim(gridDim, blockDim, meshWalls.triangles().deviceSize(), maxThreadsPerBlock))
    {
        updateTriGridCellStartEnd(triangleSpatialGrids,
        meshWalls.triangles().hashIndex(),
        meshWalls.triangles().hashValue(),
        meshWalls.triangles().index0(),
        meshWalls.triangles().index1(),
        meshWalls.triangles().index2(),
        meshWalls.globalVertices(),
        meshWalls.triangles().deviceSize(),
        gridDim,
        blockDim,
        stream);
    }

    ballTriangleInteractions.updateHistory(stream);

    if (setGPUGridBlockDim(gridDim, blockDim, balls.deviceSize(), maxThreadsPerBlock))
    {
        countBallTriangleInteractionsKernel <<<gridDim, blockDim, 0, stream>>> (balls.position(),
        balls.radius(),
        balls.inverseMass(),
        meshWalls.triangles().hashIndex(),
        meshWalls.triangles().index0(),
        meshWalls.triangles().index1(),
        meshWalls.triangles().index2(),
        meshWalls.globalVertices(),
        ballTriangleInteractionMap.countA(),
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
        int activeNumber = 0;
        cuda_copy_sync(&activeNumber, ballTriangleInteractionMap.prefixSumA() + ballTriangleInteractionMap.ASize() - 1, 1, CopyDir::D2H);
        ballTriangleInteractions.setActiveSize(static_cast<size_t>(activeNumber), stream);

        writeBallTriangleInteractionsKernel <<<gridDim, blockDim, 0, stream>>> (ballTriangleInteractions.objectPointed(),
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

        ballTriangleInteractionMap.hashInit(ballTriangleInteractions.objectPointing(), ballTriangleInteractions.activeSize(), stream);
        
        if (setGPUGridBlockDim(gridDim, blockDim, ballTriangleInteractionMap.activeHashSize(), maxThreadsPerBlock))
        {
            buildHashStartEnd(ballTriangleInteractionMap.startB(), 
            ballTriangleInteractionMap.endB(), 
            ballTriangleInteractionMap.hashIndex(), 
            ballTriangleInteractionMap.hashValue(),
            static_cast<int>(ballTriangleInteractionMap.BSize()),
            ballTriangleInteractionMap.activeHashSize(), 
            gridDim,
            blockDim,
            stream);
        }
    }
}