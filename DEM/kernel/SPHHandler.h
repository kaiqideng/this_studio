#pragma once
#include "cudaKernel/SPHNeighborSearch.h"
#include "cudaKernel/SPHIntegration.h"
#include "cudaKernel/myStruct/particle.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "cudaKernel/myStruct/interaction.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"

class SPHHandler
{
public:
    SPHHandler()
    {
        downloadSPHFlag_ = false;
    }

    ~SPHHandler() = default;

    void download(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if(downloadSPHFlag_)
        {
            size_t num0 = SPHAndGhosts_.deviceCap();
            SPHAndGhosts_.download(stream);
            size_t num1 = SPHAndGhosts_.deviceCap();
            if(num1 != num0)
            {
                SPHInteractions_.alloc(num1 * 60, stream);
                SPHInteractionMap_.alloc(num1, num1, stream);
                double cellSizeOneDim = 0.0;
                std::vector<double> h = SPHAndGhosts_.smoothLengthVector();
                if(h.size() > 0) cellSizeOneDim = *std::max_element(h.begin(), h.end()) * 2.0;
                if(cellSizeOneDim > spatialGrids_.cellSize.x 
                || cellSizeOneDim > spatialGrids_.cellSize.y 
                || cellSizeOneDim > spatialGrids_.cellSize.z)
                {
                    spatialGrids_.set(domainOrigin, domainSize, cellSizeOneDim, stream);
                }
            }
            downloadSPHFlag_ = false;
        }
    }

    void neighborSearch(const size_t maxThreads, cudaStream_t stream)
    {
        launchSPHNeighborSearch(SPHInteractions_, 
        SPHInteractionMap_, 
        SPHAndGhosts_, 
        spatialGrids_, 
        maxThreads, 
        stream);
    }

    void integration1st(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchSPH1stHalfIntegration(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        g,
        dt,
        maxThreads,
        stream);
    }

    void integration2nd(const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchSPH2ndHalfIntegration(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        dt,
        maxThreads,
        stream);
    }

    void updateBoundaryCondition(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchAdamiBoundaryCondition(SPHAndGhosts_,
        SPHInteractions_,
        SPHInteractionMap_,
        g,
        dt,
        maxThreads,
        stream);
    }
    
private:
    bool downloadSPHFlag_;

    SPH SPHAndGhosts_;
    spatialGrid spatialGrids_;

    SPHInteraction SPHInteractions_;
    interactionMap SPHInteractionMap_;
};