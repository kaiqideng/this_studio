#pragma once
#include "myUtility/myVec.h"

__device__ __forceinline__ int ParallelBondedContact(double& bondNormalForce, 
double& bondTorsionalTorque, 
double3& bondShearForce, 
double3& bondBendingTorque,
const double3 contactNormalPrev,
const double3 contactNormal,
const double3 relativeVelocityAtContact,
const double3 angularVelocityA,
const double3 angularVelocityB,
const double radiusA,
const double radiusB,
const double timeStep,
const double bondMultiplier,
const double bondElasticModulus,
const double bondStiffnessRatioNormalToShear,
const double bondTensileStrength,
const double bondCohesion,
const double bondFrictionCoefficient)
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
const double frictionCoefficient, 
const double stiffness, 
const double dampingCoefficient, 
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
const double frictionCoefficient, 
const double stiffness, 
const double dampingCoefficient, 
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

__device__ __forceinline__ void LinearContact(double3& contactForce, 
double3& contactTorque, 
double3& slidingSpring, 
double3& rollingSpring, 
double3& torsionSpring,
const double3 relativeVelocityAtContact,
const double3 relativeAngularVelocityAtContact,
const double3 contactNormal,
const double normalOverlap,
const double effectiveMass,
const double effectiveRadius,
const double timeStep,
const double normalStiffness,
const double slidingStiffness,
const double rollingStiffness,
const double torsionStiffness,
const double normalDissipation,
const double slidingDissipation,
const double rollingDissipation,
const double torsionDissipation,
const double slidingFrictionCoefficient,
const double rollingFrictionCoefficient,
const double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double normalDampingCoefficient = 2. * normalDissipation * sqrt(effectiveMass * normalStiffness);
		const double slidingDampingCoefficient = 2. * slidingDissipation * sqrt(effectiveMass * slidingStiffness);
		const double rollingDampingCoefficient = 2. * rollingDissipation * sqrt(effectiveMass * rollingStiffness);
		const double torsionDampingCoefficient = 2. * torsionDissipation * sqrt(effectiveMass * torsionStiffness);

		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, 
		slidingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		slidingFrictionCoefficient, 
		slidingStiffness, 
		slidingDampingCoefficient, 
		timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		const double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, 
		rollingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		rollingFrictionCoefficient, 
		rollingStiffness, 
		rollingDampingCoefficient, 
		timeStep);
		const double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		const double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		const double effectiveDiameter = 2 * effectiveRadius;
		const double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, 
		torsionRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		torsionFrictionCoefficient, 
		torsionStiffness, 
		torsionDampingCoefficient, 
		timeStep);
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

__device__ __forceinline__ void HertzianMindlinContact(double3& contactForce, 
double3& contactTorque, 
double3& slidingSpring, 
double3& rollingSpring, 
double3& torsionSpring,
const double3 relativeVelocityAtContact,
const double3 relativeAngularVelocityAtContact,
const double3 contactNormal,
const double normalOverlap,
const double effectiveMass,
const double effectiveRadius,
const double timeStep,
const double dissipation,
const double effectiveElasticModulus,
const double effectiveShearModulus,
const double stiffnessRatioRollingToSliding,
const double stiffnessRatioTorsionToSliding,
const double slidingFrictionCoefficient,
const double rollingFrictionCoefficient,
const double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double normalStiffness = 4. / 3. * effectiveElasticModulus * sqrt(effectiveRadius * normalOverlap);
		const double slidingStiffness = 8. * effectiveShearModulus * sqrt(effectiveRadius * normalOverlap);
		const double normalDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * normalStiffness);
		const double slidingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * slidingStiffness);

		const double rollingStiffness = slidingStiffness * stiffnessRatioRollingToSliding;
		const double torsionStiffness = slidingStiffness * stiffnessRatioTorsionToSliding;
		const double rollingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * rollingStiffness);
		const double torsionDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * torsionStiffness);

		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, 
		slidingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		slidingFrictionCoefficient, 
		slidingStiffness, 
		slidingDampingCoefficient, 
		timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		const double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, 
		rollingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		rollingFrictionCoefficient, 
		rollingStiffness, 
		rollingDampingCoefficient, 
		timeStep);
		const double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		const double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		const double effectiveDiameter = 2 * effectiveRadius;
		const double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, 
		torsionRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		torsionFrictionCoefficient, 
		torsionStiffness, 
		torsionDampingCoefficient, 
		timeStep);
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

__device__ inline bool isSphereEdgeContact(const double3& edgeP0,
const double3& edgeP1,
const double3& sphereCenter,
double sphereRadius)
{
    // Edge direction
    double3 edge = edgeP1 - edgeP0;
    double edgeLen2 = dot(edge, edge);
    if (edgeLen2 <= 1e-20) {
        // Degenerate edge -> treat as no edge contact
        return false;
    }

    // Project sphere center onto the infinite line of the edge
    double3 v = sphereCenter - edgeP0;
    double t = dot(v, edge) / edgeLen2;

    // If projection lies outside [0,1], closest point is a vertex => not edge contact
    if (t <= 0.0 || t >= 1.0) {
        return false;
    }

    // Closest point on the segment
    double3 closest = edgeP0 + edge * t;

    // Check distance to the sphere
    double3 diff = sphereCenter - closest;
    double dist2 = dot(diff, diff);
    double r2 = sphereRadius * sphereRadius;

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
const double sphereRadius,
const double3& v0,
const double3& v1,
const double3& v2,
double3& closestPoint)
{
    double3 edge01 = v1 - v0;
    double3 edge02 = v2 - v0;
    double3 v0_to_p = sphereCenter - v0;

    const double r2 = sphereRadius * sphereRadius;
    const double eps = 1e-12;

    double3 n = cross(edge01, edge02);
    double area2 = dot(n, n);
    if (area2 < 1e-20)
    {
        double3 diff0 = sphereCenter - v0;
        double3 diff1 = sphereCenter - v1;
        double3 diff2 = sphereCenter - v2;

        double d0 = dot(diff0, diff0);
        double d1 = dot(diff1, diff1);
        double d2 = dot(diff2, diff2);

        double dmin = d0;
        closestPoint = v0;

        if (d1 < dmin)
        {
            dmin = d1;
            closestPoint = v1;
        }
        if (d2 < dmin)
        {
            dmin = d2;
            closestPoint = v2;
        }

        if (dmin <= r2 + eps) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    double dot01_v0 = dot(edge01, v0_to_p);
    double dot02_v0 = dot(edge02, v0_to_p);
    if (dot01_v0 <= 0.0 && dot02_v0 <= 0.0)
    {
        closestPoint = v0;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    double3 v1_to_p = sphereCenter - v1;
    double dot01_v1 = dot(edge01, v1_to_p);
    double dot02_v1 = dot(edge02, v1_to_p);
    if (dot01_v1 >= 0.0 && dot02_v1 <= dot01_v1)
    {
        closestPoint = v1;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    double vc = dot01_v0 * dot02_v1 - dot01_v1 * dot02_v0;
    if (vc <= 0.0 && dot01_v0 >= 0.0 && dot01_v1 <= 0.0)
    {
        double t = dot01_v0 / (dot01_v0 - dot01_v1);
        closestPoint = v0 + edge01 * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    double3 v2_to_p = sphereCenter - v2;
    double dot01_v2 = dot(edge01, v2_to_p);
    double dot02_v2 = dot(edge02, v2_to_p);
    if (dot02_v2 >= 0.0 && dot01_v2 <= dot02_v2)
    {
        closestPoint = v2;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Vertex;
        return SphereTriangleContactType::None;
    }

    double vb = dot01_v2 * dot02_v0 - dot01_v0 * dot02_v2;
    if (vb <= 0.0 && dot02_v0 >= 0.0 && dot02_v2 <= 0.0)
    {
        double t = dot02_v0 / (dot02_v0 - dot02_v2);
        closestPoint = v0 + edge02 * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    double va = dot01_v1 * dot02_v2 - dot01_v2 * dot02_v1;
    if (va <= 0.0 && (dot02_v1 - dot01_v1) >= 0.0 && (dot01_v2 - dot02_v2) >= 0.0)
    {
        double t = (dot02_v1 - dot01_v1) /
        ((dot02_v1 - dot01_v1) + (dot01_v2 - dot02_v2));
        closestPoint = v1 + (v2 - v1) * t;
        double3 diff = sphereCenter - closestPoint;
        if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Edge;
        return SphereTriangleContactType::None;
    }

    double sum = va + vb + vc;
    if (fabs(sum) < 1e-20)
    {
        return SphereTriangleContactType::None;
    }

    double denom  = 1.0 / sum;
    double bary_v = vb * denom;
    double bary_w = vc * denom;

    closestPoint = v0 + edge01 * bary_v + edge02 * bary_w;
    double3 diff = sphereCenter - closestPoint;
    if (dot(diff, diff) <= r2 + eps) return SphereTriangleContactType::Face;

    return SphereTriangleContactType::None;
}

extern "C" void luanchCalculateBallContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double* radius,
double* inverseMass,
int* materialID,

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,

const double timeStep,

const size_t numInteractions,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void luanchCalculateBallDummyContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double* radius,
double* inverseMass,
int* materialID,

double3* position_dummy, 
double3* velocity_dummy, 
double3* angularVelocity_dummy, 
double* radius_dummy,
double* inverseMass_dummy,
int* materialID_dummy,

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,

const double timeStep,

const size_t numInteractions,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void luanchCalculateBondedForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque,
double* radius,
int* materialID,
int* neighborPrefixSum,

double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
int* objectPointing, 

double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
int* isBonded, 
int* objectPointed_b, 
int* objectPointing_b,

const double timeStep,

const size_t numBondedInteractions,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void luanchSumBallContactForceTorque(double3* position, 
double3* force, 
double3* torque,
int* neighborPrefixSum,
int* interactionStart, 
int* interactionEnd,

double3* contactForce,
double3* contactTorque,
double3* contactPoint,
int* interactionHashIndex,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void luanchSumBallDummyContactForceTorque(double3* position, 
double3* force, 
double3* torque,
int* neighborPrefixSum,

double3* contactForce_dummy,
double3* contactTorque_dummy,
double3* contactPoint_dummy,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void luanchCalculateBallWallContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque,
double* radius,
double* inverseMass,
int* materialID,
int* neighborPrefixSum,

double3* position_w, 
double3* velocity_w, 
double3* angularVelocity_w, 
int* materialID_w,

int* index0_t, 
int* index1_t, 
int* index2_t, 
int* wallIndex_tri,

double3* globalPosition_v, 

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring,
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,
int* cancelFlag,

const double timeStep,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);