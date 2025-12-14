#include "ballHandler.h"

class clumpHandler: 
    public ballHandler
{
public:
    clumpHandler(cudaStream_t s) :ballHandler(s)
    {
        stream_ = s;
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
    int materialID)
    {
        if(!downloadFlag_)
        {
            clumps_.upload(stream_);
            downloadFlag_ = true;
        }

        int clumpID = static_cast<int>(clumps_.hostSize());
        size_t pebbleStart = balls().hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        double volume = 0;
        for (size_t i = 0; i < points.size(); i++)
        {
            volume += 4.0 / 3.0 * pi() * pow(radius[i], 3.0);
        }
        double density_ave = 0;
        if (volume > 0.) density_ave = mass / volume;
        std::vector<double3> vel(points.size(), velocity);
        std::vector<double3> angVel(points.size(), angularVelocity);

        addCluster(points, 
        vel, 
        angVel, 
        radius, 
        density_ave, 
        materialID);

        double invMass = 0.0;
        if(mass > 1.e-20) invMass = 1.0 / mass;
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
    int materialID)
    {
        if(!downloadFlag_)
        {
            clumps_.upload(stream_);
            downloadFlag_ = true;
        }

        int clumpID = static_cast<int>(clumps_.hostSize());
        size_t pebbleStart = balls().hostSize();
        size_t pebbleEnd = pebbleStart + points.size();

        addFixedCluster(points, radius, materialID);

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

protected:
    void clumpInitialize()
    {
        downLoadClumps();
    }

    void clump1stHalfIntegration(const double3 g, const double dt, const size_t maxThreadsPerBlock)
    {
        launchClump1stHalfIntegration(clumps_, balls(), g, dt, maxThreadsPerBlock, stream_);
    }

    void clump2ndHalfIntegration(const double3 g, const double dt, const size_t maxThreadsPerBlock)
    {
        launchClump2ndHalfIntegration(clumps_, balls(), g, dt, maxThreadsPerBlock, stream_);
    }

private:
    void downLoadClumps()
    {
        if(downloadFlag_)
        {
            clumps_.download(stream_);
            downloadFlag_ = false;
        }
    }

    cudaStream_t stream_;
    bool downloadFlag_;
    clump clumps_;
};