#pragma once
#include "neighborSearch.h"
#include "myContainer/myContactModelParams.h"

__device__ __forceinline__ size_t getContactParameterArraryIndex(int mIDA, int mIDB,
                      size_t numberOfMaterials,
                      size_t d_size)
{
    const size_t N   = numberOfMaterials;
    const size_t cap = d_size;

    if (N == 0 || cap == 0) return 0;

    if (mIDA < 0 || mIDB < 0 ||
        mIDA >= static_cast<int>(N) ||
        mIDB >= static_cast<int>(N))
    {
        return cap - 1;
    }

    int i = mIDA;
    int j = mIDB;
    if (i > j) { int t = i; i = j; j = t; }

    size_t si = static_cast<size_t>(i);
    size_t sj = static_cast<size_t>(j);

    size_t idx = (si * (2 * N - si + 1)) / 2 + (sj - si);
    if (idx >= cap) idx = cap - 1;
    return idx;
}

__device__ __forceinline__ int ParallelBondedContact(double& bondNormalForce, double& bondTorsionalTorque, double3& bondShearForce, double3& bondBendingTorque,
	const double3 contactNormalPrev,
	const double3 contactNormal,
	const double3 relativeVelocityAtContact,
	const double3 angularVelocityA,
	const double3 angularVelocityB,
	const double radiusA,
	const double radiusB,
	const double timeStep,
	double bondMultiplier,
	double bondElasticModulus,
	double bondStiffnessRatioNormalToShear,
	double bondTensileStrength,
	double bondCohesion,
	double bondFrictionCoefficient)
{
	const double3 theta1 = cross(contactNormalPrev, contactNormal);
	bondShearForce = rotateVector(bondShearForce, theta1);
	bondBendingTorque = rotateVector(bondBendingTorque, theta1);
	const double3 theta2 = dot(0.5 * (angularVelocityA + angularVelocityB) * timeStep, contactNormal) * contactNormal;
	bondShearForce = rotateVector(bondShearForce, theta2);
	bondBendingTorque = rotateVector(bondBendingTorque, theta2);
	const double minRadius = radiusA < radiusB ? radiusA : radiusB;
	const double bondRadius = bondMultiplier * minRadius;
	const double bondArea = bondRadius * bondRadius * pi();// cross-section area of beam of the bond
	const double bondInertiaMoment = bondRadius * bondRadius * bondRadius * bondRadius / 4. * pi();// inertia moment
	const double bondPolarInertiaMoment = 2 * bondInertiaMoment;// polar inertia moment
	const double normalStiffnessUnitArea = bondElasticModulus / (radiusA + radiusB);
	const double shearStiffnessUnitArea = normalStiffnessUnitArea / bondStiffnessRatioNormalToShear;

	double3 normalIncrement = dot(relativeVelocityAtContact, contactNormal) * contactNormal * timeStep;
	double3 tangentialIncrement = relativeVelocityAtContact * timeStep - normalIncrement;
	bondNormalForce -= dot(normalIncrement * normalStiffnessUnitArea * bondArea, contactNormal);
	bondShearForce -= tangentialIncrement * shearStiffnessUnitArea * bondArea;
	const double3 relativeAngularVelocity = angularVelocityA - angularVelocityB;
	normalIncrement = dot(relativeAngularVelocity, contactNormal) * contactNormal * timeStep;
	tangentialIncrement = relativeAngularVelocity * timeStep - normalIncrement;
	bondTorsionalTorque -= dot(normalIncrement * shearStiffnessUnitArea * bondPolarInertiaMoment, contactNormal);
	bondBendingTorque -= tangentialIncrement * normalStiffnessUnitArea * bondInertiaMoment;

	const double maxNormalStress = -bondNormalForce / bondArea + length(bondBendingTorque) / bondInertiaMoment * bondRadius;// maximum tension stress
	const double maxShearStress = length(bondShearForce) / bondArea + fabs(bondTorsionalTorque) / bondPolarInertiaMoment * bondRadius;// maximum shear stress

	int isBonded = 1;
	if (bondTensileStrength > 0 && maxNormalStress > bondTensileStrength)
	{
		isBonded = 0;
	}
	else if (bondCohesion > 0 && maxShearStress > bondCohesion - bondFrictionCoefficient * maxNormalStress)
	{
		isBonded = 0;
	}
	return isBonded;
}

static __device__ __forceinline__ double3 integrateSlidingOrRollingSpring(const double3 springPrev, 
const double3 springVelocity, 
const double3 contactNormal, 
const double3 normalContactForce, 
double frictionCoefficient, 
double stiffness, 
double dampingCoefficient, 
const double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0)
	{
		double3 springPrev1 = springPrev - dot(springPrev, contactNormal) * contactNormal;
		double absoluteSpringPrev1 = length(springPrev1);
		if (absoluteSpringPrev1 > 1.e-10)
		{
			springPrev1 *= length(springPrev) / absoluteSpringPrev1;
		}
		spring = springPrev1 + springVelocity * timeStep;
		double3 springForce = -stiffness * spring - dampingCoefficient * springVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * springVelocity) / stiffness;
		}
	}
	return spring;
}

static __device__ __forceinline__ double3 integrateTorsionSpring(const double3 springPrev, 
const double3 torsionRelativeVelocity, 
const double3 contactNormal, 
const double3 normalContactForce, 
double frictionCoefficient, 
double stiffness, 
double dampingCoefficient, 
const double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0)
	{
		spring = dot(springPrev + torsionRelativeVelocity * timeStep, contactNormal) * contactNormal;
		double3 springForce = -stiffness * spring - dampingCoefficient * torsionRelativeVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * torsionRelativeVelocity) / stiffness;
		}
	}
	return spring;
}

__device__ __forceinline__ void LinearContact(double3& contactForce, double3& contactTorque, double3& slidingSpring, double3& rollingSpring, double3& torsionSpring,
	const double3 relativeVelocityAtContact,
	const double3 relativeAngularVelocityAtContact,
	const double3 contactNormal,
	const double normalOverlap,
	const double effectiveMass,
	const double effectiveRadius,
	const double timeStep,
	double normalStiffness,
	double slidingStiffness,
	double rollingStiffness,
	double torsionStiffness,
	double normalDissipation,
	double slidingDissipation,
	double rollingDissipation,
	double torsionDissipation,
	double slidingFrictionCoefficient,
	double rollingFrictionCoefficient,
	double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double normalDampingCoefficient = 2. * normalDissipation * sqrt(effectiveMass * normalStiffness);
		double slidingDampingCoefficient = 2. * slidingDissipation * sqrt(effectiveMass * slidingStiffness);
		double rollingDampingCoefficient = 2. * rollingDissipation * sqrt(effectiveMass * rollingStiffness);
		double torsionDampingCoefficient = 2. * torsionDissipation * sqrt(effectiveMass * torsionStiffness);

		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, slidingRelativeVelocity, contactNormal, normalContactForce, slidingFrictionCoefficient, slidingStiffness, slidingDampingCoefficient, timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		const double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, rollingRelativeVelocity, contactNormal, normalContactForce, rollingFrictionCoefficient, rollingStiffness, rollingDampingCoefficient, timeStep);
		const double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		const double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		const double effectiveDiameter = 2 * effectiveRadius;
		double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, torsionRelativeVelocity, contactNormal, normalContactForce, torsionFrictionCoefficient, torsionStiffness, torsionDampingCoefficient, timeStep);
		const double3 torsionForce = -torsionStiffness * torsionSpring - torsionDampingCoefficient * torsionRelativeVelocity;
		const double3 torsionTorque = effectiveDiameter * torsionForce;

		contactForce = normalContactForce + slidingForce;
		contactTorque = rollingTorque + torsionTorque;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
		rollingSpring = make_double3(0., 0., 0.);
		torsionSpring = make_double3(0., 0., 0.);
	}
}

__device__ __forceinline__ void HertzianMindlinContact(double3& contactForce, double3& contactTorque, double3& slidingSpring, double3& rollingSpring, double3& torsionSpring,
	const double3 relativeVelocityAtContact,
	const double3 relativeAngularVelocityAtContact,
	const double3 contactNormal,
	const double normalOverlap,
	const double effectiveMass,
	const double effectiveRadius,
	const double timeStep,
	const double dissipation,
	double effectiveElasticModulus,
	double effectiveShearModulus,
	double stiffnessRatioRollingToSliding,
	double stiffnessRatioTorsionToSliding,
	double slidingFrictionCoefficient,
	double rollingFrictionCoefficient,
	double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double normalStiffness = 4. / 3. * effectiveElasticModulus * sqrt(effectiveRadius * normalOverlap);
		double slidingStiffness = 8. * effectiveShearModulus * sqrt(effectiveRadius * normalOverlap);
		const double normalDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * normalStiffness);
		double slidingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * slidingStiffness);

		double rollingStiffness = slidingStiffness * stiffnessRatioRollingToSliding;
		double torsionStiffness = slidingStiffness * stiffnessRatioTorsionToSliding;
		double rollingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * rollingStiffness);
		double torsionDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * torsionStiffness);

		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, slidingRelativeVelocity, contactNormal, normalContactForce, slidingFrictionCoefficient, slidingStiffness, slidingDampingCoefficient, timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		const double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, rollingRelativeVelocity, contactNormal, normalContactForce, rollingFrictionCoefficient, rollingStiffness, rollingDampingCoefficient, timeStep);
		const double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		const double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		const double effectiveDiameter = 2 * effectiveRadius;
		const double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, torsionRelativeVelocity, contactNormal, normalContactForce, torsionFrictionCoefficient, torsionStiffness, torsionDampingCoefficient, timeStep);
		const double3 torsionForce = -torsionStiffness * torsionSpring - torsionDampingCoefficient * torsionRelativeVelocity;
		const double3 torsionTorque = effectiveDiameter * torsionForce;

		contactForce = normalContactForce + slidingForce;
		contactTorque = rollingTorque + torsionTorque;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
		rollingSpring = make_double3(0., 0., 0.);
		torsionSpring = make_double3(0., 0., 0.);
	}
}

__host__ __device__
inline bool isSphereEdgeContact(const double3& edgeP0,
                                const double3& edgeP1,
                                const double3& sphereCenter,
                                double         sphereRadius)
{
    // Edge direction
    double3 edge = edgeP1 - edgeP0;
    double  edgeLen2 = dot(edge, edge);
    if (edgeLen2 <= 1e-20) {
        // Degenerate edge -> treat as no edge contact
        return false;
    }

    // Project sphere center onto the infinite line of the edge
    double3 v    = sphereCenter - edgeP0;
    double  t    = dot(v, edge) / edgeLen2;

    // If projection lies outside [0,1], closest point is a vertex => not edge contact
    if (t <= 0.0 || t >= 1.0) {
        return false;
    }

    // Closest point on the segment
    double3 closest = edgeP0 + edge * t;

    // Check distance to the sphere
    double3 diff  = sphereCenter - closest;
    double  dist2 = dot(diff, diff);
    double  r2    = sphereRadius * sphereRadius;

    return dist2 <= r2;
}

enum class SphereTriangleContactType {
    None,
    Face,
    Edge,
    Vertex
};

__host__ __device__
inline SphereTriangleContactType classifySphereTriangleContact(
    const double3& sphereCenter,
    double         sphereRadius,
    const double3& v0,
    const double3& v1,
    const double3& v2,
    double3&       closestPoint  // OUT: closest point on the triangle
)
{
    double3 edge01 = v1 - v0;
    double3 edge02 = v2 - v0;
    double3 v0_to_p = sphereCenter - v0;

    const double r2 = sphereRadius * sphereRadius;

    // 1) Voronoi region of v0  -> Vertex
    double dot01_v0 = dot(edge01, v0_to_p);
    double dot02_v0 = dot(edge02, v0_to_p);
    if (dot01_v0 <= 0.0 && dot02_v0 <= 0.0) {
        closestPoint = v0;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    // 2) Voronoi region of v1 -> Vertex
    double3 v1_to_p = sphereCenter - v1;
    double dot01_v1 = dot(edge01, v1_to_p);
    double dot02_v1 = dot(edge02, v1_to_p);
    if (dot01_v1 >= 0.0 && dot02_v1 <= dot01_v1) {
        closestPoint = v1;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    // 3) Edge v0-v1 -> Edge
    double vc = dot01_v0 * dot02_v1 - dot01_v1 * dot02_v0;
    if (vc <= 0.0 && dot01_v0 >= 0.0 && dot01_v1 <= 0.0) {
        double t = dot01_v0 / (dot01_v0 - dot01_v1);
        closestPoint = v0 + edge01 * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    // 4) Voronoi region of v2 -> Vertex
    double3 v2_to_p = sphereCenter - v2;
    double dot01_v2 = dot(edge01, v2_to_p);
    double dot02_v2 = dot(edge02, v2_to_p);
    if (dot02_v2 >= 0.0 && dot01_v2 <= dot02_v2) {
        closestPoint = v2;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    // 5) Edge v0-v2 -> Edge
    double vb = dot01_v2 * dot02_v0 - dot01_v0 * dot02_v2;
    if (vb <= 0.0 && dot02_v0 >= 0.0 && dot02_v2 <= 0.0) {
        double t = dot02_v0 / (dot02_v0 - dot02_v2);
        closestPoint = v0 + edge02 * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    // 6) Edge v1-v2 -> Edge
    double va = dot01_v1 * dot02_v2 - dot01_v2 * dot02_v1;
    if (va <= 0.0 && (dot02_v1 - dot01_v1) >= 0.0 && (dot01_v2 - dot02_v2) >= 0.0) {
        double t = (dot02_v1 - dot01_v1) /
                   ((dot02_v1 - dot01_v1) + (dot01_v2 - dot02_v2));
        closestPoint = v1 + (v2 - v1) * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    // 7) Inside face -> Face
    double denom  = 1.0 / (va + vb + vc);
    double bary_v = vb * denom;
    double bary_w = vc * denom;

    closestPoint = v0 + edge01 * bary_v + edge02 * bary_w;
    double3 diff = sphereCenter - closestPoint;
    if (dot(diff, diff) <= r2) return SphereTriangleContactType::Face;

    return SphereTriangleContactType::None;
}

extern "C" void launchSolidParticleIntegrateBeforeContact(solidParticle& solidParticles, clump& clumps, const double3 gravity, const double timeStep, 
    const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchSolidParticleIntegrateAfterContact(solidParticle& solidParticles, clump& clumps, const double3 gravity, const double timeStep, 
    const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchSolidParticleInteractionCalculation(interactionSpringSystem& solidParticleInteractions, interactionBonded& bondedSolidParticleInteractions, 
    solidParticle& solidParticles, clump& clumps,
    solidContactModelParameter& contactModelParameters, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchSolidParticleInfiniteWallInteractionCalculation(interactionSpringSystem& solidParticleInfiniteWallInteractions, 
solidParticle& solidParticles, 
infiniteWall& infiniteWalls,
solidContactModelParameter& contactModelParameters, 
objectNeighborPrefix &solidParticleInfiniteWallNeighbor,
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);

extern "C" void launchInfiniteWallHalfIntegration(infiniteWall &infiniteWalls, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchTriangleWallHalfIntegration(triangleWall &triangleWalls, 
const double timeStep, 
const size_t maxThreadsPerBlock, 
cudaStream_t stream);
