#pragma once
#include "myHash.h"
#include "myUtility/myCUDA.h"
#include "myUtility/myMat.h"
#include "myUtility/myVec.h"
#include <algorithm>
#include <vector>

struct interactionBase {
private:
  std::vector<double3> h_force;
  std::vector<int> h_objectPointed;
  std::vector<int> h_objectPointing;

  size_t d_capacity{0};
  size_t d_activeNumber{0};

public:
  double3 *force{nullptr};
  int *objectPointed{nullptr};
  int *objectPointing{nullptr};

  objectHash hash;

  interactionBase() = default;
  ~interactionBase() { releaseDeviceArray(); }

  interactionBase(const interactionBase &) = delete;
  interactionBase &operator=(const interactionBase &) = delete;

  interactionBase(interactionBase &&other) noexcept {
    *this = std::move(other);
  }

  interactionBase &operator=(interactionBase &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      h_force = std::move(other.h_force);
      h_objectPointed = std::move(other.h_objectPointed);
      h_objectPointing = std::move(other.h_objectPointing);

      d_capacity = std::exchange(other.d_capacity, 0);
      d_activeNumber = std::exchange(other.d_activeNumber, 0);

      force = std::exchange(other.force, nullptr);
      objectPointed = std::exchange(other.objectPointed, nullptr);
      objectPointing = std::exchange(other.objectPointing, nullptr);

      hash = std::move(other.hash);
    }
    return *this;
  }

  size_t deviceCap() const { return d_capacity; }

  size_t activeNumber() const { return d_activeNumber; }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    if (d_capacity > 0)
      releaseDeviceArray();

    d_capacity = n;
    d_activeNumber = 0;

    hash.alloc(n, stream);

    CUDA_ALLOC(force, n, InitMode::ZERO, stream);
    CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE, stream);
    CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE, stream);
  }

  void releaseDeviceArray() {
    hash.release();

    if (force) {
      CUDA_FREE(force);
      force = nullptr;
    }
    if (objectPointed) {
      CUDA_FREE(objectPointed);
      objectPointed = nullptr;
    }
    if (objectPointing) {
      CUDA_FREE(objectPointing);
      objectPointing = nullptr;
    }

    d_capacity = 0;
    d_activeNumber = 0;
  }

  void setActiveNumber(size_t n, cudaStream_t stream) {
    if (n > d_capacity)
      allocDeviceArray(n, stream);
    d_activeNumber = n;
  }

  void setHashValue(cudaStream_t stream) const {
    hash.reset(stream);
    if (d_activeNumber > 0)
      cuda_copy(hash.value, objectPointing, d_activeNumber, CopyDir::D2D,
                stream);
  }

  std::vector<double3> getForceVectors() {
    if (d_activeNumber != h_force.size())
      h_force.resize(d_activeNumber, make_double3(0, 0, 0));
    if (d_activeNumber > 0)
      cuda_copy_sync(h_force.data(), force, d_activeNumber, CopyDir::D2H);
    return h_force;
  }

  std::vector<int> getObjectPointedVectors() {
    if (d_activeNumber != h_objectPointed.size())
      h_objectPointed.resize(d_activeNumber, -1);
    if (d_activeNumber > 0)
      cuda_copy_sync(h_objectPointed.data(), objectPointed, d_activeNumber,
                     CopyDir::D2H);
    return h_objectPointed;
  }

  std::vector<int> getObjectPointingVectors() {
    if (d_activeNumber != h_objectPointing.size())
      h_objectPointing.resize(d_activeNumber, -1);
    if (d_activeNumber > 0)
      cuda_copy_sync(h_objectPointing.data(), objectPointing, d_activeNumber,
                     CopyDir::D2H);
    return h_objectPointing;
  }
};

struct interactionSpring {
private:
  std::vector<double3> h_torque;
  std::vector<double3> h_slidingSpring;
  std::vector<double3> h_rollingSpring;
  std::vector<double3> h_torsionSpring;

  interactionBase base;

public:
  double3 *torque{nullptr};
  double3 *slidingSpring{nullptr};
  double3 *rollingSpring{nullptr};
  double3 *torsionSpring{nullptr};

  interactionSpring() = default;
  ~interactionSpring() { releaseDeviceArray(); }

  interactionSpring(const interactionSpring &) = delete;
  interactionSpring &operator=(const interactionSpring &) = delete;

  interactionSpring(interactionSpring &&other) noexcept
      : base(std::move(other.base)) {
    *this = std::move(other);
  }

  interactionSpring &operator=(interactionSpring &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      h_torque = std::move(other.h_torque);
      h_slidingSpring = std::move(other.h_slidingSpring);
      h_rollingSpring = std::move(other.h_rollingSpring);
      h_torsionSpring = std::move(other.h_torsionSpring);

      torque = std::exchange(other.torque, nullptr);
      slidingSpring = std::exchange(other.slidingSpring, nullptr);
      rollingSpring = std::exchange(other.rollingSpring, nullptr);
      torsionSpring = std::exchange(other.torsionSpring, nullptr);

      base = std::move(other.base);
    }
    return *this;
  }

  void releaseDeviceArray() {
    base.releaseDeviceArray();

    if (torque) {
      CUDA_FREE(torque);
      torque = nullptr;
    }
    if (slidingSpring) {
      CUDA_FREE(slidingSpring);
      slidingSpring = nullptr;
    }
    if (rollingSpring) {
      CUDA_FREE(rollingSpring);
      rollingSpring = nullptr;
    }
    if (torsionSpring) {
      CUDA_FREE(torsionSpring);
      torsionSpring = nullptr;
    }
  }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    if (base.deviceCap() > 0)
      releaseDeviceArray();

    base.allocDeviceArray(n, stream);

    CUDA_ALLOC(torque, n, InitMode::ZERO, stream);
    CUDA_ALLOC(slidingSpring, n, InitMode::ZERO, stream);
    CUDA_ALLOC(rollingSpring, n, InitMode::ZERO, stream);
    CUDA_ALLOC(torsionSpring, n, InitMode::ZERO, stream);
  }

  void setActiveNumber(size_t n, cudaStream_t stream) {
    if (n > base.deviceCap())
      allocDeviceArray(n, stream);

    base.setActiveNumber(n, stream);
  }

  void setHashValue(cudaStream_t stream) { base.setHashValue(stream); }

  size_t deviceCap() const { return base.deviceCap(); }

  size_t activeNumber() const { return base.activeNumber(); }

  double3 *force() { return base.force; }

  int *objectPointed() { return base.objectPointed; }

  int *objectPointing() { return base.objectPointing; }

  objectHash &hash() { return base.hash; }

  std::vector<double3> getForceVectors() { return base.getForceVectors(); }

  std::vector<double3> getTorqueVectors() {
    size_t activeNumber = base.activeNumber();
    if (activeNumber != h_torque.size())
      h_torque.resize(activeNumber, make_double3(0, 0, 0));
    if (activeNumber > 0)
      cuda_copy_sync(h_torque.data(), torque, activeNumber, CopyDir::D2H);
    return h_torque;
  }

  std::vector<double3> getSlidingSpringVectors() {
    size_t activeNumber = base.activeNumber();
    if (activeNumber != h_slidingSpring.size())
      h_slidingSpring.resize(activeNumber, make_double3(0, 0, 0));
    if (activeNumber > 0)
      cuda_copy_sync(h_slidingSpring.data(), slidingSpring, activeNumber,
                     CopyDir::D2H);
    return h_slidingSpring;
  }

  std::vector<double3> getRollingSpringVectors() {
    size_t activeNumber = base.activeNumber();
    if (activeNumber != h_rollingSpring.size())
      h_rollingSpring.resize(activeNumber, make_double3(0, 0, 0));
    if (activeNumber > 0)
      cuda_copy_sync(h_rollingSpring.data(), rollingSpring, activeNumber,
                     CopyDir::D2H);
    return h_rollingSpring;
  }

  std::vector<double3> getTorsionSpringVectors() {
    size_t activeNumber = base.activeNumber();
    if (activeNumber != h_torsionSpring.size())
      h_torsionSpring.resize(activeNumber, make_double3(0, 0, 0));
    if (activeNumber > 0)
      cuda_copy_sync(h_torsionSpring.data(), torsionSpring, activeNumber,
                     CopyDir::D2H);
    return h_torsionSpring;
  }

  std::vector<int> getObjectPointedVectors() {
    return base.getObjectPointedVectors();
  }

  std::vector<int> getObjectPointingVectors() {
    return base.getObjectPointingVectors();
  }
};

struct interactionSpringHistory {
private:
  interactionBase base;

public:
  double3 *slidingSpring{nullptr};
  double3 *rollingSpring{nullptr};
  double3 *torsionSpring{nullptr};

  interactionSpringHistory() = default;

  ~interactionSpringHistory() { releaseDeviceArray(); }

  interactionSpringHistory(const interactionSpringHistory &) = delete;
  interactionSpringHistory &
  operator=(const interactionSpringHistory &) = delete;

  interactionSpringHistory(interactionSpringHistory &&other) noexcept {
    *this = std::move(other);
  }

  interactionSpringHistory &
  operator=(interactionSpringHistory &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      slidingSpring = std::exchange(other.slidingSpring, nullptr);
      rollingSpring = std::exchange(other.rollingSpring, nullptr);
      torsionSpring = std::exchange(other.torsionSpring, nullptr);

      base = std::move(other.base);
    }
    return *this;
  }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    if (base.deviceCap() > 0)
      releaseDeviceArray();

    base.allocDeviceArray(n, stream);

    CUDA_ALLOC(slidingSpring, n, InitMode::ZERO, stream);
    CUDA_ALLOC(rollingSpring, n, InitMode::ZERO, stream);
    CUDA_ALLOC(torsionSpring, n, InitMode::ZERO, stream);
  }

  void releaseDeviceArray() {
    if (slidingSpring) {
      CUDA_FREE(slidingSpring);
      slidingSpring = nullptr;
    }
    if (rollingSpring) {
      CUDA_FREE(rollingSpring);
      rollingSpring = nullptr;
    }
    if (torsionSpring) {
      CUDA_FREE(torsionSpring);
      torsionSpring = nullptr;
    }

    base.releaseDeviceArray();
  }

  void setActiveNumber(size_t n, cudaStream_t stream) {
    if (n > base.deviceCap())
      allocDeviceArray(n, stream);

    base.setActiveNumber(n, stream);
  }

  double3 *force() { return base.force; }

  int *objectPointed() { return base.objectPointed; }

  int *objectPointing() { return base.objectPointing; }

  objectHash &hash() { return base.hash; }
};

struct interactionSpringSystem {
  interactionSpring current;
  interactionSpringHistory history;

  interactionSpringSystem() = default;
  ~interactionSpringSystem() { releaseDeviceArray(); }

  interactionSpringSystem(const interactionSpringSystem &) = delete;
  interactionSpringSystem &operator=(const interactionSpringSystem &) = delete;

  interactionSpringSystem(interactionSpringSystem &&other) noexcept {
    *this = std::move(other);
  }

  interactionSpringSystem &operator=(interactionSpringSystem &&other) noexcept {
    if (this != &other) {
      releaseDeviceArray();

      current = std::move(other.current);
      history = std::move(other.history);
    }
    return *this;
  }

  void releaseDeviceArray() {
    current.releaseDeviceArray();
    history.releaseDeviceArray();
  }

  void allocDeviceArray(size_t n, cudaStream_t stream) {
    current.allocDeviceArray(n, stream);
    history.allocDeviceArray(n, stream);
  }

  void setCurrentActiveNumber(size_t n, cudaStream_t stream) {
    current.setActiveNumber(n, stream);
  }

  void setHashValue(cudaStream_t stream) { current.setHashValue(stream); }

  void recordCurrentInteractionSpring(cudaStream_t stream) {
    size_t activeNumber = current.activeNumber();
    history.setActiveNumber(activeNumber, stream);

    if (activeNumber <= 0)
      return;
    cuda_copy(history.objectPointed(), current.objectPointed(), activeNumber,
              CopyDir::D2D, stream);
    cuda_copy(history.objectPointing(), current.objectPointing(), activeNumber,
              CopyDir::D2D, stream);
    cuda_copy(history.slidingSpring, current.slidingSpring, activeNumber,
              CopyDir::D2D, stream);
    cuda_copy(history.rollingSpring, current.rollingSpring, activeNumber,
              CopyDir::D2D, stream);
    cuda_copy(history.torsionSpring, current.torsionSpring, activeNumber,
              CopyDir::D2D, stream);
  }

  size_t getActiveNumber() const { return current.activeNumber(); }

  std::vector<double3> getForceVectors() { return current.getForceVectors(); }

  std::vector<double3> getTorqueVectors() { return current.getTorqueVectors(); }

  std::vector<double3> getSlidingSpringVectors() {
    return current.getSlidingSpringVectors();
  }

  std::vector<double3> getRollingSpringVectors() {
    return current.getRollingSpringVectors();
  }

  std::vector<double3> getTorsionSpringVectors() {
    return current.getTorsionSpringVectors();
  }

  std::vector<int> getObjectPointedVectors() {
    return current.getObjectPointedVectors();
  }

  std::vector<int> getObjectPointingVectors() {
    return current.getObjectPointingVectors();
  }
};

struct interactionBonded {
private:
  std::vector<double3> h_contactNormal;
  std::vector<double3> h_shearForce;
  std::vector<double3> h_bendingTorque;
  std::vector<double> h_normalForce;
  std::vector<double> h_torsionTorque;

  std::vector<int> h_objectPointed;
  std::vector<int> h_objectPointing;
  std::vector<int> h_isBonded;

  size_t d_size{0};

  void release() {
    if (contactNormal) {
      CUDA_FREE(contactNormal);
      contactNormal = nullptr;
    }
    if (shearForce) {
      CUDA_FREE(shearForce);
      shearForce = nullptr;
    }
    if (bendingTorque) {
      CUDA_FREE(bendingTorque);
      bendingTorque = nullptr;
    }
    if (normalForce) {
      CUDA_FREE(normalForce);
      normalForce = nullptr;
    }
    if (torsionTorque) {
      CUDA_FREE(torsionTorque);
      torsionTorque = nullptr;
    }

    if (objectPointed) {
      CUDA_FREE(objectPointed);
      objectPointed = nullptr;
    }
    if (objectPointing) {
      CUDA_FREE(objectPointing);
      objectPointing = nullptr;
    }
    if (isBonded) {
      CUDA_FREE(isBonded);
      isBonded = nullptr;
    }

    d_size = 0;
  }

  void alloc(size_t n, cudaStream_t stream) {
    if (d_size > 0)
      release();

    d_size = n;

    CUDA_ALLOC(contactNormal, n, InitMode::ZERO, stream);
    CUDA_ALLOC(shearForce, n, InitMode::ZERO, stream);
    CUDA_ALLOC(bendingTorque, n, InitMode::ZERO, stream);
    CUDA_ALLOC(normalForce, n, InitMode::ZERO, stream);
    CUDA_ALLOC(torsionTorque, n, InitMode::ZERO, stream);

    CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE, stream);
    CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE, stream);
    CUDA_ALLOC(isBonded, n, InitMode::ZERO, stream);
  }

  void downLoad(cudaStream_t stream) {
    size_t n = h_isBonded.size();
    if (n > d_size)
      alloc(n, stream);

    cuda_copy(contactNormal, h_contactNormal.data(), n, CopyDir::H2D, stream);
    cuda_copy(shearForce, h_shearForce.data(), n, CopyDir::H2D, stream);
    cuda_copy(bendingTorque, h_bendingTorque.data(), n, CopyDir::H2D, stream);
    cuda_copy(normalForce, h_normalForce.data(), n, CopyDir::H2D, stream);
    cuda_copy(torsionTorque, h_torsionTorque.data(), n, CopyDir::H2D, stream);

    cuda_copy(objectPointed, h_objectPointed.data(), d_size, CopyDir::H2D,
              stream);
    cuda_copy(objectPointing, h_objectPointing.data(), d_size, CopyDir::H2D,
              stream);
    cuda_copy(isBonded, h_isBonded.data(), d_size, CopyDir::H2D, stream);
  }

  void upLoad() {
    if (d_size == 0 || d_size > h_isBonded.size())
      return;

    cuda_copy_sync(h_contactNormal.data(), contactNormal, d_size, CopyDir::D2H);
    cuda_copy_sync(h_shearForce.data(), shearForce, d_size, CopyDir::D2H);
    cuda_copy_sync(h_bendingTorque.data(), bendingTorque, d_size, CopyDir::D2H);
    cuda_copy_sync(h_normalForce.data(), normalForce, d_size, CopyDir::D2H);
    cuda_copy_sync(h_torsionTorque.data(), torsionTorque, d_size, CopyDir::D2H);

    cuda_copy_sync(h_objectPointed.data(), objectPointed, d_size, CopyDir::D2H);
    cuda_copy_sync(h_objectPointing.data(), objectPointing, d_size,
                   CopyDir::D2H);
    cuda_copy_sync(h_isBonded.data(), isBonded, d_size, CopyDir::H2D);
  }

public:
  double3 *contactNormal{nullptr};
  double3 *shearForce{nullptr};
  double3 *bendingTorque{nullptr};
  double *normalForce{nullptr};
  double *torsionTorque{nullptr};

  int *objectPointed{nullptr};
  int *objectPointing{nullptr};
  int *isBonded{nullptr};

  interactionBonded() = default;

  ~interactionBonded() { release(); }

  interactionBonded(const interactionBonded &) = delete;
  interactionBonded &operator=(const interactionBonded &) = delete;

  interactionBonded(interactionBonded &&other) noexcept {
    *this = std::move(other);
  }

  interactionBonded &operator=(interactionBonded &&other) noexcept {
    if (this != &other) {
      release();

      h_contactNormal = std::move(other.h_contactNormal);
      h_shearForce = std::move(other.h_shearForce);
      h_bendingTorque = std::move(other.h_bendingTorque);
      h_normalForce = std::move(other.h_normalForce);
      h_shearForce = std::move(other.h_shearForce);

      h_objectPointed = std::move(other.h_objectPointed);
      h_objectPointing = std::move(other.h_objectPointing);
      h_isBonded = std::move(other.h_isBonded);

      d_size = std::exchange(other.d_size, 0);

      contactNormal = std::exchange(other.contactNormal, nullptr);
      shearForce = std::exchange(other.shearForce, nullptr);
      bendingTorque = std::exchange(other.bendingTorque, nullptr);
      normalForce = std::exchange(other.normalForce, nullptr);
      torsionTorque = std::exchange(other.torsionTorque, nullptr);

      objectPointed = std::exchange(other.objectPointed, nullptr);
      objectPointing = std::exchange(other.objectPointing, nullptr);
      isBonded = std::exchange(other.isBonded, nullptr);
    }
    return *this;
  }

  size_t size() const { return d_size; }

  void add(const std::vector<int> &ob0, const std::vector<int> &ob1,
           const std::vector<double3> &p, cudaStream_t stream) {
    if (ob0.size() != ob1.size())
      return;

    upLoad();

    for (size_t i = 0; i < ob0.size(); ++i) {
      int i0 = ob0[i];
      int i1 = ob1[i];

      // basic validity checks
      if (i0 < 0 || i1 < 0)
        continue;
      if (static_cast<size_t>(i0) >= p.size() ||
          static_cast<size_t>(i1) >= p.size())
        continue;
      if (i0 == i1)
        continue; // no self bond

      int a = i0;
      int b = i1;
      if (a > b)
        std::swap(a, b);

      // check if bond (a, b) already exists
      bool found = false;
      for (size_t j = 0; j < h_objectPointed.size(); ++j) {
        if (h_objectPointed[j] == a && h_objectPointing[j] == b) {
          found = true;
          break;
        }
      }
      if (found)
        continue;

      // create new bond
      h_objectPointed.push_back(a);
      h_objectPointing.push_back(b);

      const double3 n = p[a] - p[b];
      h_contactNormal.push_back(normalize(n));

      h_shearForce.push_back(make_double3(0.0, 0.0, 0.0));
      h_bendingTorque.push_back(make_double3(0.0, 0.0, 0.0));
      h_normalForce.push_back(0.0);
      h_torsionTorque.push_back(0.0);
      h_isBonded.push_back(1);
    }

    downLoad(stream);
  }
};