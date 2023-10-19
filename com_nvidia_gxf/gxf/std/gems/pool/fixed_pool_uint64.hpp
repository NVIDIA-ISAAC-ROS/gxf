/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_GEMS_POOL_FIXED_POOL_UINT64_HPP
#define NVIDIA_GXF_STD_GEMS_POOL_FIXED_POOL_UINT64_HPP

#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "common/expected.hpp"

namespace nvidia {
namespace gxf {

// An efficient fixed-size pool of uint64 indices.
class FixedPoolUint64 {
 public:
  // Error codes used by this class.
  enum class Error {
    kUnderflow,
    kOverflow,
    kInvalidUid,
    kOutOfMemory,
  };

  // Expected type used by this class.
  template <typename T>
  using expected_t = nvidia::Expected<T, Error>;
  // Unexpected type used by this class.
  using unexpected_t = nvidia::Unexpected<Error>;

  FixedPoolUint64()
      : size_(0), next_(0), available_(nullptr), indices_(nullptr) {}

  ~FixedPoolUint64() {
    if (size_ != 0) {
      delete[] available_;
      delete[] indices_;
    }
  }

  FixedPoolUint64(FixedPoolUint64&&) = delete;
  FixedPoolUint64(const FixedPoolUint64&) = delete;
  FixedPoolUint64& operator=(FixedPoolUint64&&) = delete;
  FixedPoolUint64& operator=(const FixedPoolUint64&) = delete;

  expected_t<void> allocate(uint64_t size) {
    if (size_ != 0) {
      delete[] available_;
      delete[] indices_;
    }

    size_ = size;
    next_ = 0;

    if (size_ > 0) {
      available_ = new(std::nothrow) uint64_t[size_];
      if (available_ == nullptr) {
        size_ = 0;
        return unexpected_t{Error::kOutOfMemory};
      }
      indices_ = new(std::nothrow) uint64_t[size_];
      if (indices_ == nullptr) {
        delete[] available_;
        available_ = nullptr;
        size_ = 0;
        return unexpected_t{Error::kOutOfMemory};
      }
    } else {
      available_ = nullptr;
      indices_ = nullptr;
    }

    for (uint64_t i = 0; i < size_; i++) {
      available_[i] = i;
      indices_[i] = i;
    }

    return expected_t<void>{};
  }

  bool is_available() const {
    return next_ < size_;
  }

  size_t capacity() const {
    return size_;
  }

  size_t size() const {
    return size_ - next_;
  }

  expected_t<uint64_t> pop() {
    if (next_ == size_) {
      // no more items available
      return unexpected_t{Error::kUnderflow};
    }
    return available_[next_++];
  }

  expected_t<void> push(uint64_t item) {
    if (item >= size_) {
      // not part of the stack
      return unexpected_t{Error::kOverflow};
    }

    // get the index at which this item is stored
    const uint64_t item_index = indices_[item];
    if (item_index >= next_) {
      // already free
      return unexpected_t{Error::kInvalidUid};
    }

    // make item available by swapping with the currently first unavailable
    next_--;  // next - 1 is the index of the first available at the moment
    const uint64_t other = available_[next_];  // the value at that index
    std::swap(available_[item_index], available_[next_]);
    std::swap(indices_[item], indices_[other]);

    return expected_t<void>{};
  }

 private:
  uint64_t size_;
  uint64_t next_;
  uint64_t* available_;
  uint64_t* indices_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
