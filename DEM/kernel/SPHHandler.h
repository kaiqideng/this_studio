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
        downloadVirtualParticleFlag_ = false;
        maximumAbsluteVelocity = 0.0;
    }

    ~SPHHandler() = default;

    void download(const double3 domainOrigin, const double3 domainSize, cudaStream_t stream)
    {
        if(downloadSPHFlag_)
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
        }

        if(downloadVirtualParticleFlag_)
        {
            virtualParticles_.download(stream);
        }

        if(downloadVirtualParticleFlag_ || downloadSPHFlag_)
        {
            SPHVirtualInteractions_.alloc(SPHs_.deviceSize() * 60, stream);
            SPHVirtualInteractionMap_.alloc(SPHs_.deviceSize(), virtualParticles_.deviceSize(), stream);
        }

        downloadSPHFlag_ = false;
        downloadVirtualParticleFlag_ = false;
    }

    void neighborSearch(const size_t maxThreads, cudaStream_t stream)
    {
        launchSPHNeighborSearch(SPHInteractions_, 
        SPHInteractionMap_, 
        SPHs_, 
        SPHVirtualInteractions_,
        SPHVirtualInteractionMap_,
        virtualParticles_,
        spatialGrids_, 
        maxThreads, 
        stream);
    }

    void integration(const double3 g, const double dt, const size_t maxThreads, cudaStream_t stream)
    {
        launchSPHIntegration(SPHs_, 
        SPHInteractions_, 
        SPHInteractionMap_, 
        virtualParticles_, 
        SPHVirtualInteractions_, 
        SPHVirtualInteractionMap_, 
        maximumAbsluteVelocity, 
        g, 
        dt, 
        maxThreads, 
        stream);
    }
    
private:
    bool downloadSPHFlag_;
    bool downloadVirtualParticleFlag_;
    SPH SPHs_;
    virtualParticle virtualParticles_;
    spatialGrid spatialGrids_;
    double maximumAbsluteVelocity;

    SPHInteraction SPHInteractions_;
    interactionMap SPHInteractionMap_;
    SPHInteraction SPHVirtualInteractions_;
    interactionMap SPHVirtualInteractionMap_;
};