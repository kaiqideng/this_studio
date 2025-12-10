#pragma once
#include "myContainer/myParticle.h"
#include "myContainer/mySpatialGrid.h"
#include "myContainer/myUtility/myFileEdit.h"
#include "myContainer/myUtility/myMat.h"
#include <vector>

class wallHandler
{
public:
    wallHandler(cudaStream_t s)
    {

    }
    ~wallHandler() = default;
};