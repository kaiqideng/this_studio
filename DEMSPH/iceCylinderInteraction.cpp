#include "kernel/DEMSolver.h"
#include "externalForceTorque.h"
#include "pointCloudGeneration.h"

inline double waterSurfaceHeightForStaticSphere(double R,
                                               double rhoBall,
                                               double rhoWater)
{
    if (!(R > 0.0) || !(rhoWater > 0.0) || !(rhoBall >= 0.0))
        return R; // simple fallback

    const double f = rhoBall / rhoWater; // required submerged volume fraction

    if (f <= 0.0) return -R;  // no buoyancy needed -> essentially not submerged
    if (f >= 1.0) return  R;  // sinks or neutral -> return R (your rule)

    // Submerged volume fraction for plane z = h:
    // f(h) = (R + h)^2 * (2R - h) / (4 R^3), monotone on [-R, R]
    auto submergedFraction = [&](double h) -> double
    {
        const double b = R + h;
        return (b * b) * (2.0 * R - h) / (4.0 * R * R * R);
    };

    // Bisection on h in [-R, R]
    double lo = -R, hi = R;
    for (int it = 0; it < 80; ++it)
    {
        const double mid = 0.5 * (lo + hi);
        const double fm  = submergedFraction(mid);
        if (fm < f) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

class problem:
    public DEMSolver
{
public:
    double waterLevel = 0.;

    problem(): DEMSolver(0) {}

    void addExternalForceTorque(const size_t iStep, const double time) override
    {
        launchAddBuoyancyDrag(getBall().force(),
        getBall().position(),
        getBall().velocity(),
        getBall().radius(),
        getBall().inverseMass(),
        1000.,
        0.1,
        make_double3(0., 0., -9.81),
        make_double3(-0.4, 0., 0.),
        waterLevel,
        getBall().deviceSize(),
        getBall().gridDim(),
        getBall().blockDim(),
        0);
    }

    bool addInitialCondition() override
    {
        std::vector<int> obj0 = getBallPairPointed();
        std::vector<int> obj1 = getBallPairPointing();
        std::vector<double3> pos = getBall().positionHostCopy();
        addBondedInteraction(obj0, obj1, pos);
        return true;
    }
};

int main()
{
    double tc = 0.0002;
    double r = 0.1;
    double den = 900.;
    double m = r * r * r * pi() * 4. / 3. * den;
    double kn = 0.5 * m * pi() * pi() / tc / tc;
    double E = kn * 2. * r / (r * r * pi());

    problem test;
    test.waterLevel = waterSurfaceHeightForStaticSphere(r, den, 1000);

    test.setBondedPair(0, 0, 1., E, 2.6, 0.5e6, 0.5e6, 0.1);
    test.setLinearPair(0, 0, kn, kn / 2.6, 0., 0., 0.1, 0.1, 0., 0., 0.1, 0., 0.);
    test.setLinearPair(0, 1, kn, kn / 2.6, 0., 0., 0.1, 0.1, 0., 0., 0.1, 0., 0.);

    std::vector<double3> points;
    HexPBC2D pbc;
    generateHexPointCloud(make_double3(0., 0., 0.), 
    make_double3(200 * r, 200 * r, 0.), 
    2 * r, 
    false, 
    true, 
    points, 
    pbc);

    TriangleMesh mesh = makeCylinderMesh(make_double3(0., 0., 0.), 
    make_double3(0., 0., 1.), 
    1.,
    6.,
    36,
    36);

    for (size_t i = 0; i < points.size(); i++)
    {
        test.addBall(points[i], 
        make_double3(-0.4, 0., 0.), 
        make_double3(0., 0., 0.), 
        r, 
        den, 
        0);
    }

    test.addMeshWall(mesh.vertices, 
    mesh.tri0, 
    mesh.tri1, 
    mesh.tri2, 
    make_double3(-1.1, 100 * r, 0.), 
    make_double3(0., 0., 0.), 
    make_double3(0., 0., 0.), 
    1);

    test.solve(make_double3(-pbc.max.x, pbc.min.y, -3.), 
    make_double3(pbc.max.x, pbc.max.y, 3.), 
    make_double3(0., 0., -9.81), 
    tc / 50., 
    50., 
    1000, 
    "iceCylinderInteraction");
}
