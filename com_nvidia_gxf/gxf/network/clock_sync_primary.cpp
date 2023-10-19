/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/clock_sync_primary.hpp"

#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t ClockSyncPrimary::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(tx_timestamp_, "tx_timestamp", "Outgoing timestamp",
    "The outgoing timestamp channel");
  result &= registrar->parameter(clock_, "clock", "Application clock",
    "Handle to application's clock component");
  return ToResultCode(result);
}

gxf_result_t ClockSyncPrimary::tick() {
  // Create entity to be published
  Expected<Entity> maybe_entity = Entity::New(context());
  if (!maybe_entity) {
    return maybe_entity.error();
  }
  Entity& entity = maybe_entity.value();

  // Add timestamp to entity
  Expected<Handle<Timestamp>> maybe_timestamp =
      entity.add<Timestamp>("timestamp");
  if (!maybe_timestamp) {
    return maybe_timestamp.error();
  }
  Timestamp& timestamp = *maybe_timestamp.value();

  // Set timestamp acqtime to latest clock timestamp, and pubtime to
  // when this codelet prepares the message.
  timestamp.acqtime = clock_->timestamp();
  timestamp.pubtime = getExecutionTimestamp();
  GXF_LOG_DEBUG("Issued primary clock message with timestamp %ldns.", timestamp.acqtime);

  return ToResultCode(tx_timestamp_->publish(entity));
}

}  // namespace gxf
}  // namespace nvidia
