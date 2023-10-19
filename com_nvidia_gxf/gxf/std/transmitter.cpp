/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/transmitter.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

Expected<void> Transmitter::publish(const Entity& other) {
  auto timestamp = other.get<Timestamp>("timestamp");
  if (timestamp) {
    // "Deprecation warning (will be removed in the next release): Timestamp already present in "
    // "the message. Use publish(message, acq_time) instead. to add timestamps");
  }
  return ExpectedOrCode(publish_abi(other.eid()));
}

Expected<void> Transmitter::publish(Entity& other, const int64_t acq_timestamp) {
  auto timestamp = other.get<Timestamp>("timestamp");
  if (!timestamp) {
    timestamp = other.add<Timestamp>("timestamp");
    if (!timestamp) {
      GXF_LOG_ERROR("Failure creating Timestamp component for message.");
      return Unexpected{timestamp.error()};
    }
  }
  timestamp.value()->pubtime = 0;
  timestamp.value()->acqtime = acq_timestamp;
  return ExpectedOrCode(publish_abi(other.eid()));
}

size_t Transmitter::back_size() {
  return back_size_abi();
}

Expected<void> Transmitter::sync() {
  return ExpectedOrCode(sync_abi());
}

Expected<void> Transmitter::sync_io() {
  return ExpectedOrCode(sync_io_abi());
}

}  // namespace gxf
}  // namespace nvidia
