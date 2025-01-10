/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/benchmark/benchmark_allocator.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

BenchmarkAllocator::~BenchmarkAllocator() {}

gxf_result_t BenchmarkAllocator::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(storage_type_, "storage_type", "", "",
    static_cast<int32_t>(MemoryStorageType::kHost));
  result &= registrar->parameter(clock_, "clock", "Clock Component",
    "Used to add timestamp in the messages");
  result &= registrar->parameter(output_, "output", "Output",
    "Transmitter channel publishing messages");
  result &= registrar->parameter(
    block_size_, "block_size", "Block size", "The size of one block of memory in byte");
  result &= registrar->parameter(
    number_of_blocks_, "number_of_blocks", "Number of blocks",
    "The total number of blocks which are allocated by the pool");
  result &= registrar->parameter(
    number_of_iterations_, "number_of_iterations", "Number of iterations",
    "The total number of iterations for which allocation and free will be called");
  result &= registrar->parameter(
    allocator_, "allocator", "Memory allocator",
    "Memory allocator for benchmarking");
  result &= registrar->parameter(in_, "in", "input", "The channel for incoming messages.");
  return ToResultCode(result);
}

gxf_result_t BenchmarkAllocator::initialize() {
  allocation_count_ = 0;
  iteration_count_ = 0;
  allocation_type_ = AllocationType::kAllocate;

  return GXF_SUCCESS;
}

gxf_result_t BenchmarkAllocator::tick() {
  auto message = in_->receive();
  if (!message) {
    return message.error();
  }
  if (iteration_count_ < number_of_iterations_) {
    auto message = Entity::New(context());
    if (!message) {
      GXF_LOG_DEBUG("Unable to create message entity");
      return ToResultCode(message);
    }
    const int64_t acqtime = clock_.get()->timestamp();
    if ((allocation_count_ < number_of_blocks_) &&
        (allocation_type_ == AllocationType::kAllocate)) {
      if (pointer_.size() != static_cast<size_t>(allocation_count_)) {
        GXF_LOG_ERROR("Mismatch in the no of buffer stored v/s allocation count"
                      "Allocation count = %d, buffer size = %zu",
                      allocation_count_, pointer_.size());
        return GXF_FAILURE;
      }
      const auto maybe = UNWRAP_OR_RETURN(
        allocator_->allocate(block_size_, static_cast<MemoryStorageType>(storage_type_.get())));
      auto allocation_type = message.value().add<int32_t>("Allocate");
      if (!allocation_type) {
        GXF_LOG_ERROR("Unable to add allocation_type in the message");
        return ToResultCode(allocation_type);
      }
      *allocation_type.value() = static_cast<int32_t>(AllocationType::kAllocate);
      pointer_.push_back(maybe);
      allocation_count_++;
    } else {
      allocation_type_ = AllocationType::kFree;
      if (!pointer_.size()) {
        GXF_LOG_ERROR("Invalid state. Buffer pointer not available");
        return GXF_FAILURE;
      }
      auto result = allocator_->free(pointer_.front());
      if (!result) {
        return ToResultCode(result);
      }
      pointer_.erase(pointer_.begin());
      auto allocation_type = message.value().add<int32_t>("Free");
      if (!allocation_type) {
        GXF_LOG_ERROR("Unable to add allocation_type in the message");
        return ToResultCode(allocation_type);
      }
      *allocation_type.value() = static_cast<int32_t>(AllocationType::kFree);
      allocation_count_--;
    }

    const uint64_t pubtime = clock_.get()->timestamp();
    auto timestamp = message.value().add<Timestamp>();
    if (!timestamp) {
      GXF_LOG_ERROR("Unable to add timestamp in the message");
      return ToResultCode(timestamp);
    }
    timestamp.value()->acqtime = acqtime;
    timestamp.value()->pubtime = pubtime;
    auto result = output_->publish(message.value());
    if (!result) {
      GXF_LOG_ERROR("Failed to publish message");
      return ToResultCode(result);
    }

    if ((allocation_count_ == 0) && (allocation_type_ == AllocationType::kFree)) {
      iteration_count_++;
      allocation_type_ = AllocationType::kAllocate;
      pointer_.clear();
    }
  } else {
    iteration_count_ = 0;
  }

  return GXF_SUCCESS;
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
