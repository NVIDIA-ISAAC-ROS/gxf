/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/broadcast.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t Broadcast::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(source_, "source", "Source channel", "");
  result &= registrar->parameter(mode_, "mode", "Broadcast Mode",
                                 "The broadcast mode. Can be Broadcast or RoundRobin.",
                                 BroadcastMode::kBroadcast);
  return ToResultCode(result);
}

gxf_result_t Broadcast::start() {
  auto result = entity().findAll<Transmitter>().assign_to(tx_list_);
  if (!result) {
    return ToResultCode(result);
  }
  if (tx_list_.size() == 0) {
    GXF_LOG_ERROR("No Transmitter instance found on the entity");
    return GXF_ARGUMENT_NULL;
  }
  return GXF_SUCCESS;
}

gxf_result_t Broadcast::tick() {
  const auto message = source_->receive();
  if (!message) {
    return message.error();
  }

  switch (static_cast<BroadcastMode>(mode_)) {
    case BroadcastMode::kBroadcast: {
      for (auto transmitter : tx_list_) {
        if (!transmitter) {
          GXF_LOG_ERROR("Found a null handle to a Transmitter");
          return GXF_FAILURE;
        }
        auto result = transmitter.value()->publish(message.value());
        if (!result) {
          return ToResultCode(result);
        }
      }
      break;
    }
    case BroadcastMode::kRoundRobin: {
      auto transmitter = tx_list_[round_robin_tx_index_++ % tx_list_.size()];
      if (!transmitter) {
        GXF_LOG_ERROR("Found a null handle to a Transmitter");
        return GXF_FAILURE;
      }
      auto result = transmitter.value()->publish(message.value());
      if (!result) {
        return ToResultCode(result);
      }
      break;
    }
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  return GXF_SUCCESS;
}

gxf_result_t Broadcast::stop() {
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
