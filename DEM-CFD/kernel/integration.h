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

extern "C" void launchSolidParticleIntegrateBeforeContact(solidParticle& solidParticles, clump& clumps, const double3 gravity, const double timeStep, 
    const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchSolidParticleIntegrateAfterContact(solidParticle& solidParticles, clump& clumps, const double3 gravity, const double timeStep, 
    const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchSolidParticleInteractionCalculation(interactionSpringSystem& solidParticleInteractions, interactionBonded& bondedSolidParticleInteractions, 
    solidParticle& solidParticles, clump& clumps,
    solidContactModelParameter& contactModelParameters, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream);

extern "C" void launchInfiniteWallHalfIntegration(infiniteWall &infiniteWalls, const double timeStep, const size_t maxThreadsPerBlock, cudaStream_t stream);