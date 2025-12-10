#pragma once
#include "myHash.h"
#include "myUtility/myCUDA.h"
#include "myUtility/myMat.h"
#include <algorithm>
#include <vector>

struct particleBase {
private:
  std::vector<double3> h_position;
  std::vector<double3> h_velocity;
  std::vector<double> h_effectiveRadii;

  size_t d_size{0};

public:
  double3 *position{nullptr};
  double3 *velocity{nullptr};
  double *effectiveRadii{nullptr};

  particleBase() = default;
  ~particleBase() { releaseDeviceArray(); }
  particleBase(const particleBase &) = delete;
  particleBase &operator=(const particleBase &) = delete;

  particleBase(particleBase &&other) noexcept { *this = std::move(other); }

  particleBase &operator=(particleBase &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      h_position = std::move(other.h_position);
      h_velocity = std::move(other.h_velocity);
      h_effectiveRadii = std::move(other.h_effectiveRadii);

      position = std::exchange(other.position, nullptr);
      velocity = std::exchange(other.velocity, nullptr);
      effectiveRadii = std::exchange(other.effectiveRadii, nullptr);

      d_size = std::exchange(other.d_size, 0);
    }
    return *this;
  }

  size_t hostSize() { return h_position.size(); }

  size_t deviceSize() const { return d_size; }

  void releaseDeviceArray() {
    if (position) {
      CUDA_FREE(position);
      position = nullptr;
    }
    if (velocity) {
      CUDA_FREE(velocity);
      velocity = nullptr;
    }
    if (effectiveRadii) {
      CUDA_FREE(effectiveRadii);
      effectiveRadii = nullptr;
    }

    d_size = 0;
  }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    if (d_size > 0)
      releaseDeviceArray();

    d_size = n;

    CUDA_ALLOC(position, n, InitMode::ZERO, stream);
    CUDA_ALLOC(velocity, n, InitMode::ZERO, stream);
    CUDA_ALLOC(effectiveRadii, n, InitMode::ZERO, stream);
  }

  void add(const double3 &newPosition, const double3 &newVelocity,
           double newEffectiveRadii) {
    h_position.push_back(newPosition);
    h_velocity.push_back(newVelocity);
    h_effectiveRadii.push_back(newEffectiveRadii);
  }

  void remove(size_t index) {
    if (index >= h_position.size())
      return;

    h_position.erase(h_position.begin() + index);
    h_velocity.erase(h_velocity.begin() + index);
    h_effectiveRadii.erase(h_effectiveRadii.begin() + index);
  }

  void download(cudaStream_t stream) {
    size_t h_size = hostSize();

    if (h_size != d_size)
      allocDeviceArray(h_size, stream);

    if (h_size > 0) {
      cuda_copy(position, h_position.data(), h_size, CopyDir::H2D, stream);
      cuda_copy(velocity, h_velocity.data(), h_size, CopyDir::H2D, stream);
      cuda_copy(effectiveRadii, h_effectiveRadii.data(), h_size, CopyDir::H2D,
                stream);
    }
  }

  void upload(cudaStream_t stream) {
    const size_t n = d_size;
    if (n == 0)
      return;

    if (hostSize() != n)
      return;

    cuda_copy(h_position.data(), position, n, CopyDir::D2H, stream);
    cuda_copy(h_velocity.data(), velocity, n, CopyDir::D2H, stream);
  }

  void setPositionVectors(const std::vector<double3> &newPosition) {
    if (newPosition.size() <= hostSize()) {
      std::copy(newPosition.begin(), newPosition.end(), h_position.begin());
      if (newPosition.size() <= deviceSize())
        cuda_copy_sync(position, newPosition.data(), newPosition.size(),
                       CopyDir::H2D);
    }
  }

  void setVelocityVectors(const std::vector<double3> &newVelocity) {
    if (newVelocity.size() <= hostSize()) {
      std::copy(newVelocity.begin(), newVelocity.end(), h_velocity.begin());
      if (newVelocity.size() <= deviceSize())
        cuda_copy_sync(velocity, newVelocity.data(), newVelocity.size(),
                       CopyDir::H2D);
    }
  }

  void setRadiiVectors(const std::vector<double> &newRadii) {
    if (newRadii.size() <= hostSize()) {
      std::copy(newRadii.begin(), newRadii.end(), h_effectiveRadii.begin());
      if (newRadii.size() <= deviceSize())
        cuda_copy_sync(effectiveRadii, newRadii.data(), newRadii.size(),
                       CopyDir::H2D);
    }
  }

  std::vector<double3> getPositionVectors() {
    if (d_size <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_position.data(), position, d_size, CopyDir::D2H);
    return h_position;
  }

  std::vector<double3> getVelocityVectors() {
    if (d_size <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_velocity.data(), velocity, d_size, CopyDir::D2H);
    return h_velocity;
  }

  const std::vector<double> getEffectiveRadii() const {
    return h_effectiveRadii;
  }
};

struct solidParticle {
private:
  std::vector<double3> h_force;
  std::vector<double3> h_torque;
  std::vector<double3> h_angularVelocity;

  std::vector<double> h_radius;
  std::vector<double> h_inverseMass;

  std::vector<int> h_materialID;

  std::vector<int> h_clumpID;

  particleBase base;

  void releaseDeviceArray() {
    base.releaseDeviceArray();
    hash.release();
    neighbor.release();
    interactionIndexRange.release();

    if (force) {
      CUDA_FREE(force);
      force = nullptr;
    }
    if (torque) {
      CUDA_FREE(torque);
      torque = nullptr;
    }
    if (angularVelocity) {
      CUDA_FREE(angularVelocity);
      angularVelocity = nullptr;
    }
    if (radius) {
      CUDA_FREE(radius);
      radius = nullptr;
    }
    if (inverseMass) {
      CUDA_FREE(inverseMass);
      inverseMass = nullptr;
    }
    if (materialID) {
      CUDA_FREE(materialID);
      materialID = nullptr;
    }
    if (clumpID) {
      CUDA_FREE(clumpID);
      clumpID = nullptr;
    }
  }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    if (base.deviceSize() > 0)
      releaseDeviceArray();

    base.allocDeviceArray(n, stream);
    hash.alloc(n, stream);
    neighbor.alloc(n, stream);
    interactionIndexRange.alloc(n, stream);

    CUDA_ALLOC(force, n, InitMode::ZERO, stream);
    CUDA_ALLOC(torque, n, InitMode::ZERO, stream);
    CUDA_ALLOC(angularVelocity, n, InitMode::ZERO, stream);
    CUDA_ALLOC(radius, n, InitMode::ZERO, stream);
    CUDA_ALLOC(inverseMass, n, InitMode::ZERO, stream);
    CUDA_ALLOC(materialID, n, InitMode::ZERO, stream);
    CUDA_ALLOC(clumpID, n, InitMode::NEG_ONE, stream);
  }

public:
  double3 *force{nullptr};
  double3 *torque{nullptr};
  double3 *angularVelocity{nullptr};
  double *radius{nullptr};
  double *inverseMass{nullptr};
  int *materialID{nullptr};
  int *clumpID{nullptr};

  objectHash hash;
  objectNeighborPrefix neighbor;
  sortedHashValueIndex interactionIndexRange;

  solidParticle() = default;
  ~solidParticle() { releaseDeviceArray(); }
  solidParticle(const solidParticle &) = delete;
  solidParticle &operator=(const solidParticle &) = delete;
  solidParticle(solidParticle &&other) noexcept { *this = std::move(other); }

  solidParticle &operator=(solidParticle &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      h_force = std::move(other.h_force);
      h_torque = std::move(other.h_torque);
      h_angularVelocity = std::move(other.h_angularVelocity);
      h_radius = std::move(other.h_radius);
      h_inverseMass = std::move(other.h_inverseMass);
      h_materialID = std::move(other.h_materialID);
      h_clumpID = std::move(other.h_clumpID);

      force = std::exchange(other.force, nullptr);
      torque = std::exchange(other.torque, nullptr);
      angularVelocity = std::exchange(other.angularVelocity, nullptr);
      radius = std::exchange(other.radius, nullptr);
      inverseMass = std::exchange(other.inverseMass, nullptr);
      materialID = std::exchange(other.materialID, nullptr);
      clumpID = std::exchange(other.clumpID, nullptr);

      base = std::move(other.base);
      hash = std::move(other.hash);
      neighbor = std::move(other.neighbor);
      interactionIndexRange = std::move(other.interactionIndexRange);
    }
    return *this;
  }

  size_t hostSize() { return base.hostSize(); }

  size_t deviceSize() const { return base.deviceSize(); }

  void add(const double3 &pos, const double3 &vel,
           double r, // r is physical radius
           const double3 &F, const double3 &T, const double3 &aVel, double invM,
           int mID, int clumpID) {
    base.add(pos, vel, 1.1 * r);

    h_force.push_back(F);
    h_torque.push_back(T);
    h_angularVelocity.push_back(aVel);
    h_radius.push_back(r);
    h_inverseMass.push_back(invM);
    h_materialID.push_back(mID);
    h_clumpID.push_back(clumpID);
  }

  void remove(size_t index) {
    if (index >= base.hostSize())
      return;

    base.remove(index);

    h_force.erase(h_force.begin() + index);
    h_torque.erase(h_torque.begin() + index);
    h_angularVelocity.erase(h_angularVelocity.begin() + index);
    h_radius.erase(h_radius.begin() + index);
    h_inverseMass.erase(h_inverseMass.begin() + index);
    h_materialID.erase(h_materialID.begin() + index);
    h_clumpID.erase(h_clumpID.begin() + index);
  }

  void download(cudaStream_t stream) {
    size_t n = base.hostSize();
    if (n != base.deviceSize())
      allocDeviceArray(n, stream);

    if (n > 0) {
      base.download(stream);
      cuda_copy(force, h_force.data(), n, CopyDir::H2D, stream);
      cuda_copy(torque, h_torque.data(), n, CopyDir::H2D, stream);
      cuda_copy(angularVelocity, h_angularVelocity.data(), n, CopyDir::H2D,
                stream);
      cuda_copy(radius, h_radius.data(), n, CopyDir::H2D, stream);
      cuda_copy(inverseMass, h_inverseMass.data(), n, CopyDir::H2D, stream);
      cuda_copy(materialID, h_materialID.data(), n, CopyDir::H2D, stream);
      cuda_copy(clumpID, h_clumpID.data(), n, CopyDir::H2D, stream);
    }
  }

  void upload(cudaStream_t stream) {
    const size_t n = base.deviceSize();
    if (n == 0)
      return;

    if (hostSize() != n)
      return;

    base.upload(stream);

    cuda_copy(h_force.data(), force, n, CopyDir::D2H, stream);
    cuda_copy(h_torque.data(), torque, n, CopyDir::D2H, stream);
    cuda_copy(h_angularVelocity.data(), angularVelocity, n, CopyDir::D2H,
              stream);
  }

  void setPositionVectors(const std::vector<double3> &newPosition) {
    base.setPositionVectors(newPosition);
  }

  void setVelocityVectors(const std::vector<double3> &newVelocity) {
    base.setVelocityVectors(newVelocity);
  }

  void setRadiiVectors(const std::vector<double> &newRadii) {
    base.setRadiiVectors(newRadii);
  }

  void
  setAngularVelocityVectors(const std::vector<double3> &newAngularVelocity) {
    if (newAngularVelocity.size() <= hostSize()) {
      std::copy(newAngularVelocity.begin(), newAngularVelocity.end(),
                h_angularVelocity.begin());
      if (newAngularVelocity.size() <= deviceSize())
        cuda_copy_sync(angularVelocity, newAngularVelocity.data(),
                       newAngularVelocity.size(), CopyDir::H2D);
    }
  }

  void setForceVectors(const std::vector<double3> &newForce) {
    if (newForce.size() <= hostSize()) {
      std::copy(newForce.begin(), newForce.end(), h_force.begin());
      if (newForce.size() <= deviceSize())
        cuda_copy_sync(force, newForce.data(), newForce.size(), CopyDir::H2D);
    }
  }

  void setTorqueVectors(const std::vector<double3> &newTorque) {
    if (newTorque.size() <= hostSize()) {
      std::copy(newTorque.begin(), newTorque.end(), h_torque.begin());
      if (newTorque.size() <= deviceSize())
        cuda_copy_sync(torque, newTorque.data(), newTorque.size(),
                       CopyDir::H2D);
    }
  }

  std::vector<double3> getPositionVectors() {
    return base.getPositionVectors();
  }

  std::vector<double3> getVelocityVectors() {
    return base.getVelocityVectors();
  }

  const std::vector<double> getEffectiveRadii() const {
    return base.getEffectiveRadii();
  }

  std::vector<double3> getAngularVelocityVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_angularVelocity.data(), angularVelocity, deviceSize(),
                     CopyDir::D2H);
    return h_angularVelocity;
  }

  std::vector<double3> getForceVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_force.data(), force, deviceSize(), CopyDir::D2H);
    return h_force;
  }

  std::vector<double3> getTorqueVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_torque.data(), torque, deviceSize(), CopyDir::D2H);
    return h_torque;
  }

  const std::vector<double> getRadiusVectors() const { return h_radius; }

  const std::vector<int> getMaterialIDVectors() const { return h_materialID; }

  const std::vector<int> getClumpIDVectors() const { return h_clumpID; }

  double3 *position() { return base.position; }

  double3 *velocity() { return base.velocity; }

  double *effectiveRadii() { return base.effectiveRadii; }

  void clearForceTorque(cudaStream_t stream) {
    CUDA_CHECK(
        cudaMemsetAsync(force, 0.0, deviceSize() * sizeof(double3), stream));
    CUDA_CHECK(
        cudaMemsetAsync(torque, 0.0, deviceSize() * sizeof(double3), stream));
  }
};

struct clump {
private:
  std::vector<double3> h_force;
  std::vector<double3> h_torque;
  std::vector<double3> h_angularVelocity;
  std::vector<quaternion> h_orientation;
  std::vector<symMatrix> h_inverseInertiaTensor;
  std::vector<double> h_inverseMass;
  std::vector<int> h_pebbleStartIndex;
  std::vector<int> h_pebbleEndIndex;

  particleBase base;

  void releaseDeviceArray() {
    base.releaseDeviceArray();

    if (force) {
      CUDA_FREE(force);
      force = nullptr;
    }
    if (torque) {
      CUDA_FREE(torque);
      torque = nullptr;
    }
    if (angularVelocity) {
      CUDA_FREE(angularVelocity);
      angularVelocity = nullptr;
    }
    if (orientation) {
      CUDA_FREE(orientation);
      orientation = nullptr;
    }
    if (inverseInertiaTensor) {
      CUDA_FREE(inverseInertiaTensor);
      inverseInertiaTensor = nullptr;
    }
    if (inverseMass) {
      CUDA_FREE(inverseMass);
      inverseMass = nullptr;
    }
    if (pebbleStartIndex) {
      CUDA_FREE(pebbleStartIndex);
      pebbleStartIndex = nullptr;
    }
    if (pebbleEndIndex) {
      CUDA_FREE(pebbleEndIndex);
      pebbleEndIndex = nullptr;
    }
  }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    if (base.deviceSize() > 0)
      releaseDeviceArray();

    base.allocDeviceArray(n, stream);

    CUDA_ALLOC(force, n, InitMode::ZERO, stream);
    CUDA_ALLOC(torque, n, InitMode::ZERO, stream);
    CUDA_ALLOC(angularVelocity, n, InitMode::ZERO, stream);
    CUDA_ALLOC(orientation, n, InitMode::ZERO, stream);
    CUDA_ALLOC(inverseInertiaTensor, n, InitMode::ZERO, stream);
    CUDA_ALLOC(inverseMass, n, InitMode::ZERO, stream);
    CUDA_ALLOC(pebbleStartIndex, n, InitMode::NEG_ONE, stream);
    CUDA_ALLOC(pebbleEndIndex, n, InitMode::NEG_ONE, stream);
  }

public:
  double3 *force{nullptr};
  double3 *torque{nullptr};
  double3 *angularVelocity{nullptr};
  quaternion *orientation{nullptr};
  symMatrix *inverseInertiaTensor{nullptr};
  double *inverseMass{nullptr};
  int *pebbleStartIndex{nullptr};
  int *pebbleEndIndex{nullptr};

  clump() = default;
  ~clump() { releaseDeviceArray(); }
  clump(const clump &) = delete;
  clump &operator=(const clump &) = delete;

  clump(clump &&other) noexcept { *this = std::move(other); }

  clump &operator=(clump &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      h_force = std::move(other.h_force);
      h_torque = std::move(other.h_torque);
      h_angularVelocity = std::move(other.h_angularVelocity);
      h_orientation = std::move(other.h_orientation);
      h_inverseInertiaTensor = std::move(other.h_inverseInertiaTensor);
      h_inverseMass = std::move(other.h_inverseMass);
      h_pebbleStartIndex = std::move(other.h_pebbleStartIndex);
      h_pebbleEndIndex = std::move(other.h_pebbleEndIndex);

      force = std::exchange(other.force, nullptr);
      torque = std::exchange(other.torque, nullptr);
      angularVelocity = std::exchange(other.angularVelocity, nullptr);
      orientation = std::exchange(other.orientation, nullptr);
      inverseInertiaTensor = std::exchange(other.inverseInertiaTensor, nullptr);
      inverseMass = std::exchange(other.inverseMass, nullptr);
      pebbleStartIndex = std::exchange(other.pebbleStartIndex, nullptr);
      pebbleEndIndex = std::exchange(other.pebbleEndIndex, nullptr);

      base = std::move(other.base);
    }
    return *this;
  }

  size_t hostSize() { return base.hostSize(); }

  size_t deviceSize() const { return base.deviceSize(); }

  void add(const double3 &pos, const double3 &vel, const double3 &aVel,
           const quaternion &orient, const symMatrix &invI, double invM,
           size_t startI, size_t endI) {
    base.add(pos, vel, 0.0);

    h_force.push_back(make_double3(0, 0, 0));
    h_torque.push_back(make_double3(0, 0, 0));
    h_angularVelocity.push_back(aVel);
    h_orientation.push_back(orient);
    h_inverseInertiaTensor.push_back(invI);
    h_inverseMass.push_back(invM);
    h_pebbleStartIndex.push_back(startI);
    h_pebbleEndIndex.push_back(endI);
  }

  void download(cudaStream_t stream) {
    size_t n = base.hostSize();

    if (n != base.deviceSize())
      allocDeviceArray(n, stream);

    if (n > 0) {
      base.download(stream);

      cuda_copy(force, h_force.data(), n, CopyDir::H2D, stream);
      cuda_copy(torque, h_torque.data(), n, CopyDir::H2D, stream);
      cuda_copy(angularVelocity, h_angularVelocity.data(), n, CopyDir::H2D,
                stream);
      cuda_copy(orientation, h_orientation.data(), n, CopyDir::H2D, stream);
      cuda_copy(inverseInertiaTensor, h_inverseInertiaTensor.data(), n,
                CopyDir::H2D, stream);
      cuda_copy(inverseMass, h_inverseMass.data(), n, CopyDir::H2D, stream);
      cuda_copy(pebbleStartIndex, h_pebbleStartIndex.data(), n, CopyDir::H2D,
                stream);
      cuda_copy(pebbleEndIndex, h_pebbleEndIndex.data(), n, CopyDir::H2D,
                stream);
    }
  }

  void upload(cudaStream_t stream) {
    size_t n = base.deviceSize();
    if (n == 0)
      return;

    if (hostSize() != n)
      return;

    base.upload(stream);

    cuda_copy(h_force.data(), force, n, CopyDir::D2H, stream);
    cuda_copy(h_torque.data(), torque, n, CopyDir::D2H, stream);
    cuda_copy(h_angularVelocity.data(), angularVelocity, n, CopyDir::D2H,
              stream);
    cuda_copy(h_orientation.data(), orientation, n, CopyDir::D2H, stream);
  }

  void setPositionVectors(const std::vector<double3> &newPosition) {
    base.setPositionVectors(newPosition);
  }

  void setVelocityVectors(const std::vector<double3> &newVelocity,
                          cudaStream_t stream) {
    base.setVelocityVectors(newVelocity);
  }

  void
  setAngularVelocityVectors(const std::vector<double3> &newAngularVelocity) {
    if (newAngularVelocity.size() <= hostSize()) {
      std::copy(newAngularVelocity.begin(), newAngularVelocity.end(),
                h_angularVelocity.begin());
      if (newAngularVelocity.size() <= deviceSize())
        cuda_copy_sync(angularVelocity, newAngularVelocity.data(),
                       newAngularVelocity.size(), CopyDir::H2D);
    }
  }

  void setForceVectors(const std::vector<double3> &newForce) {
    if (newForce.size() <= hostSize()) {
      std::copy(newForce.begin(), newForce.end(), h_force.begin());
      if (newForce.size() <= deviceSize())
        cuda_copy_sync(force, newForce.data(), newForce.size(), CopyDir::H2D);
    }
  }

  void setTorqueVectors(const std::vector<double3> &newTorque) {
    if (newTorque.size() <= hostSize()) {
      std::copy(newTorque.begin(), newTorque.end(), h_torque.begin());
      if (newTorque.size() <= deviceSize())
        cuda_copy_sync(torque, newTorque.data(), newTorque.size(),
                       CopyDir::H2D);
    }
  }

  std::vector<double3> getPositionVectors() {
    return base.getPositionVectors();
  }

  std::vector<double3> getVelocityVectors() {
    return base.getVelocityVectors();
  }

  std::vector<double3> getAngularVelocityVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_angularVelocity.data(), angularVelocity, deviceSize(),
                     CopyDir::D2H);
    return h_angularVelocity;
  }

  std::vector<quaternion> getOrientationVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_orientation.data(), orientation, deviceSize(),
                     CopyDir::D2H);
    return h_orientation;
  }

  std::vector<double3> getForceVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_force.data(), force, deviceSize(), CopyDir::D2H);
    return h_force;
  }

  std::vector<double3> getTorqueVectors() {
    if (deviceSize() <= hostSize() && deviceSize() > 0)
      cuda_copy_sync(h_torque.data(), torque, deviceSize(), CopyDir::D2H);
    return h_torque;
  }

  double3 *position() { return base.position; }

  double3 *velocity() { return base.velocity; }

  void clearForceTorque(cudaStream_t stream) {
    CUDA_CHECK(
        cudaMemsetAsync(force, 0.0, deviceSize() * sizeof(double3), stream));
    CUDA_CHECK(
        cudaMemsetAsync(torque, 0.0, deviceSize() * sizeof(double3), stream));
  }
};