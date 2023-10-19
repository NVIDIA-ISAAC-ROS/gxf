/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/clock_sync_secondary.hpp"

#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t ClockSyncSecondary::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(rx_timestamp_, "rx_timestamp", "Incoming timestamp",
    "The incoming timestamp channel");
  result &= registrar->parameter(synthetic_clock_, "synthetic_clock",
    "Application's synthetic clock", "Handle to application's synthetic clock component");
  return ToResultCode(result);
}

gxf_result_t ClockSyncSecondary::tick() {
  // Pull entity from receiver
  Expected<Entity> maybe_entity = rx_timestamp_.get()->receive();
  if (!maybe_entity) {
    return maybe_entity.error();
  }

  // Retrieve primary clock timestamp from entity
  Expected<Handle<Timestamp>> maybe_timestamp_handle =
      maybe_entity->get<Timestamp>("timestamp");
  if (!maybe_timestamp_handle) {
    return maybe_timestamp_handle.error();
  }
  Timestamp& timestamp = *maybe_timestamp_handle.value();
  const auto primary_clock_timestamp = timestamp.acqtime;
  GXF_LOG_DEBUG("Received primary clock message with timestamp %ldns.", primary_clock_timestamp);

  // Advance our application's synthetic clock to the received timestamp
  const auto result = synthetic_clock_->advanceTo(primary_clock_timestamp);
  if (!result) {
      GXF_LOG_ERROR("Failed to advance synthetic clock to primary clock timestamp %ldns.",
        primary_clock_timestamp);
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
