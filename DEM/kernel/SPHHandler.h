#pragma once
#include "cudaKernel/SPHNeighborSearch.h"
#include "cudaKernel/myStruct/particle.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "cudaKernel/myStruct/interaction.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"

class SPHHandler
{
public:
    SPHHandler()
    {
        downLoadFlag_ = false;
    }

    ~SPHHandler() = default;

    void download(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if(downLoadFlag_)
        {
            size_t numSPHs0 = SPHs_.deviceSize();
            SPHs_.download(stream);
            size_t numSPHs1 = SPHs_.deviceSize();
            if(numSPHs1 != numSPHs0)
            {
                SPHInteractions_.alloc(numSPHs1 * 60, stream);
                SPHInteractionMap_.alloc(numSPHs1, numSPHs1, stream);
                double cellSizeOneDim = 0.0;
                std::vector<double> h = SPHs_.smoothLengthVector();
                if(h.size() > 0) cellSizeOneDim = *std::max_element(h.begin(), h.end()) * 2.0;
                if(cellSizeOneDim > spatialGrids_.cellSize.x 
                || cellSizeOneDim > spatialGrids_.cellSize.y 
                || cellSizeOneDim > spatialGrids_.cellSize.z)
                {
                    spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream);
                }
            }
            downLoadFlag_ = false;
        }
    }

    void neighborSearch(const size_t maxThreads, cudaStream_t stream)
    {
        launchSPHNeighborSearch(SPHInteractions_, 
        SPHInteractionMap_, 
        SPHs_, 
        spatialGrids_, 
        maxThreads, 
        stream);
    }

    void integration1st(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {

    }

    void integration2nd(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        
    }

private:
    bool downLoadFlag_;
    SPH SPHs_;
    spatialGrid spatialGrids_;

    SPHInteraction SPHInteractions_;
    interactionMap SPHInteractionMap_;
};