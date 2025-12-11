#pragma once
#include "myUtility/myMat.h"
#include "myUtility/myHostDeviceArray1D.h"
#include "myUtility/myQua.h"
#include <vector_functions.h>

struct infiniteWall
{
private:
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> axis_;
    HostDeviceArray1D<double>  axisAngularVelocity_;
    HostDeviceArray1D<double>  radius_;
    HostDeviceArray1D<int>     materialID_;

public:
    infiniteWall() = default;
    ~infiniteWall() = default;
    infiniteWall(const infiniteWall&) = delete;
    infiniteWall& operator=(const infiniteWall&) = delete;
    infiniteWall(infiniteWall&&) noexcept = default;
    infiniteWall& operator=(infiniteWall&&) noexcept = default;

    size_t hostSize() const
    {
        return position_.hostSize();
    }

    size_t deviceSize() const
    {
        return position_.deviceSize();
    }

    void addPlane(const double3& pos,
             const double3& vel,
             const double3& n,
             int materialID)
    {
        position_.addHostData(pos);
        velocity_.addHostData(vel);
        axis_.addHostData(n);
        axisAngularVelocity_.addHostData(0.0);
        radius_.addHostData(0.0);
        materialID_.addHostData(materialID);
    }

    void addCylinder(const double3& pos,
             const double3& vel,
             const double3& axis,
             const double& angVel,
             double r,
             int materialID)
    {
        position_.addHostData(pos);
        velocity_.addHostData(vel);
        axis_.addHostData(axis);
        axisAngularVelocity_.addHostData(angVel);
        radius_.addHostData(r);
        materialID_.addHostData(materialID);
    }

    void remove(size_t index)
    {
        position_.removeHostData(index);
        velocity_.removeHostData(index);
        axis_.removeHostData(index);
        axisAngularVelocity_.removeHostData(index);
        radius_.removeHostData(index);
        materialID_.removeHostData(index);
    }

    void clearHost()
    {
        position_.clearHostData();
        velocity_.clearHostData();
        axis_.clearHostData();
        axisAngularVelocity_.clearHostData();
        radius_.clearHostData();
        materialID_.clearHostData();
    }

    void download(cudaStream_t stream)
    {
        position_.download(stream);
        velocity_.download(stream);
        axis_.download(stream);
        axisAngularVelocity_.download(stream);
        radius_.download(stream);
        materialID_.download(stream);
    }

    void upload(cudaStream_t stream)
    {
        position_.upload(stream);
        velocity_.upload(stream);
        axis_.upload(stream);
        axisAngularVelocity_.upload(stream);
        radius_.upload(stream);
        materialID_.upload(stream);
    }

    double3* position()
    {
        return position_.d_ptr;
    }

    double3* velocity()
    {
        return velocity_.d_ptr;
    }

    double3* axis()
    {
        return axis_.d_ptr;
    }

    double* axisAngularVelocity()
    {
        return axisAngularVelocity_.d_ptr;
    }

    double* radius()
    {
        return radius_.d_ptr;
    }

    int* materialID()
    {
        return materialID_.d_ptr;
    }

    const std::vector<double3> getPositionHost()
    {
        return position_.getHostData();
    }

    const std::vector<double3> getVelocityHost()
    {
        return velocity_.getHostData();
    }

    const std::vector<double3> getAxisHost()
    {
        return axis_.getHostData();
    }

    const std::vector<double> getAxisAngularVelocityHost()
    {
        return axisAngularVelocity_.getHostData();
    }

    const std::vector<double> getRadiusHost()
    {
        return radius_.getHostData();
    }

    const std::vector<int> getMaterialIDHost()
    {
        return materialID_.getHostData();
    }
};


struct triangleWall
{
private:
    HostDeviceArray1D<double3> position_;
    HostDeviceArray1D<double3> velocity_;
    HostDeviceArray1D<double3> angularVelocity_;
    HostDeviceArray1D<quaternion> orientation_;
    HostDeviceArray1D<int>     materialID_;

public:
    triangleWall() = default;
    ~triangleWall() = default;
    triangleWall(const triangleWall&) = delete;
    triangleWall& operator=(const triangleWall&) = delete;
    triangleWall(triangleWall&&) noexcept = default;
    triangleWall& operator=(triangleWall&&) noexcept = default;

    size_t hostSize() const
    {
        return position_.hostSize();
    }

    size_t deviceSize() const
    {
        return position_.deviceSize();
    }

    void download(cudaStream_t stream)
    {
        
    }
};