#include "ballHandler.h"
#include "cudaKernel/myStruct/myUtility/myVec.h"
#include "cudaKernel/myStruct/particle.h"

class clumpHandler
{
public:
    clumpHandler()
    {
        downloadFlag_ = false;
    }

    ~clumpHandler() = default;

    void addClump(std::vector<double3> points, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    double3 velocity, 
    double3 angularVelocity, 
    double mass, 
    symMatrix inertiaTensor, 
    int materialID,
    ballHandler& bH,
    cudaStream_t stream)
    {
        if (!downloadFlag_)
        {
            clumps_.upload(stream);
            downloadFlag_ = true;
        }

        int clumpID = static_cast<int>(clumps_.hostSize());
        size_t pebbleStart = bH.getBalls().hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        std::vector<double3> vel(points.size(), velocity);
        std::vector<double3> angVel(points.size(), angularVelocity);
        double volume = 0, invMass = 0.0;
        if (mass > 1.e-20)
        {
            invMass = 1.0 / mass;
            for (size_t i = 0; i < points.size(); i++)
            {
                volume += 4.0 / 3.0 * pi() * pow(radius[i], 3.0);
                vel[i] = vel[i] + cross(angularVelocity, points[i] - centroidPosition);
            }
        }
        double density_ave = 0;
        if (volume > 0.) density_ave = mass / volume;

        bH.addCluster(points, 
        vel, 
        angVel, 
        radius, 
        density_ave, 
        materialID,
        stream,
        clumpID);

        clumps_.addHost(centroidPosition, 
        velocity, 
        angularVelocity, 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0,0.0,0.0,0.0), 
        inverse(inertiaTensor), 
        invMass, 
        materialID,
        pebbleStart, 
        pebbleEnd);
    }

    void addFixedClump(std::vector<double3> points, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    int materialID,
    ballHandler& bH,
    cudaStream_t stream)
    {
        if (!downloadFlag_)
        {
            clumps_.upload(stream);
            downloadFlag_ = true;
        }

        int clumpID = static_cast<int>(clumps_.hostSize());
        size_t pebbleStart = bH.getBalls().hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        bH.addFixedCluster(points, 
        radius, 
        materialID, 
        stream, 
        clumpID);

        clumps_.addHost(centroidPosition, 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0,0.0,0.0,0.0), 
        make_symMatrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 
        0.0, 
        materialID,
        pebbleStart, 
        pebbleEnd);
    }

    clump &getClumps() { return clumps_; }

    void download(cudaStream_t stream)
    {
        if (downloadFlag_)
        {
            clumps_.download(stream);
            downloadFlag_ = false;
        }
    }

    void integration1st(ball& balls, const double3 g, const double dt, const size_t gridDim, const size_t blockDim, cudaStream_t stream)
	{
        launchClump1stHalfIntegration(clumps_, 
        balls, 
        g, 
        dt, 
        gridDim,
        blockDim,
        stream);
	}

    void integration2nd(ball& balls, const double3 g, const double dt,  const size_t gridDim, const size_t blockDim, cudaStream_t stream)
	{
        launchClump2ndHalfIntegration(clumps_, 
        balls, 
        g, 
        dt, 
        gridDim,
        blockDim,
        stream);
	}

private:
    bool downloadFlag_;
    clump clumps_;
};