/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <algorithm>
#include <vector>

#include "gxf/core/expected_macro.hpp"
#include "gxf/std/synchronization.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t Synchronization::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      inputs_, "inputs", "Inputs",
      "All the inputs for synchronization, number of inputs must match that of the outputs.");
  result &= registrar->parameter(
      outputs_, "outputs", "Outputs",
      "All the outputs for synchronization, number of outputs must match that of the inputs.");
  result &= registrar->parameter(
      sync_threshold_, "sync_threshold", "Synchronization threshold (ns)",
      "Synchronization threshold in nanoseconds. "
      "Messages will not be synchronized if timestamp difference is above the threshold. "
      "By default, timestamps should be identical for synchronization (default threshold = 0). "
      "Synchronization threshold will only work if maximum timestamp variation is much less "
      "than minimal delta between timestamps of subsequent messages in any input.",
      static_cast<int64_t>(0));
  return ToResultCode(result);
}

gxf_result_t Synchronization::start() {
  if (inputs_.get().size() != outputs_.get().size()) {
    GXF_LOG_ERROR("Number of inputs for synchronization must match the number of outputs");
    return GXF_FAILURE;
  }
  if (inputs_.get().size() <= 1) {
    GXF_LOG_ERROR("Number of inputs/outputs should be more than 1");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t Synchronization::tick() {
  /* Check if all input queues have messages */
  for (const auto& rx : inputs_.get()) {
    if (rx->size() == 0) {
      /* Not all the inputs have messages, unexpected. */
      GXF_LOG_ERROR("Not all the inputs have messages for synchronization!");
      return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
    }
  }

  /* First try reading the timestamps of available messages from all the inputs */
  std::vector<std::vector<int64_t>> acq_times;
  for (const auto& rx : inputs_.get()) {
    std::vector<int64_t> ack_times_per_receiver;
    for (size_t index = rx->size(); index > 0; index--) {
      auto msg_result = rx->peek(index - 1);
      if (msg_result) {
        auto timestamp_components = msg_result->findAllHeap<Timestamp>();
        if (!timestamp_components) {
          return ToResultCode(timestamp_components);
        }
        if (0 == timestamp_components->size()) {
          GXF_LOG_ERROR("No timestamp found from the input message");
          return GXF_ENTITY_COMPONENT_NOT_FOUND;
        }
        ack_times_per_receiver.push_back(timestamp_components->front().value()->acqtime);
      }
    }
    acq_times.push_back(ack_times_per_receiver);
  }

  /* find latest timestamp we're going to pick from the fastest moving queue*/
  int64_t latest = 0;
  for (const auto& timestamps : acq_times) {
    if (timestamps.back() > latest) { latest = timestamps.back(); }
  }
  GXF_LOG_DEBUG("Candidate timestamp for syncing: %zd", latest);
  /* check if in all the monitored receivers there are messages that are within the threshold of
     the latest timestamp, e.g., if there is a sync point */
  uint32_t synchronized = 0;
  auto threshold = sync_threshold_.get();
  auto lower = latest;
  auto upper = latest;
  for (const auto& timestamps : acq_times) {
    auto it = std::find_if(timestamps.begin(), timestamps.end(),
                           [lower, upper, threshold](int64_t acq_time) {
                             return lower + threshold >= acq_time && acq_time >= upper - threshold;
                           });
    if (it != timestamps.end()) {
      synchronized++;
      lower = std::min(lower, *it);
      upper = std::max(upper, *it);
    }
  }
  const bool can_send = synchronized == inputs_.get().size();

  // poll the message queues based on the gathered timestamp information
  // the assumption here is the timestamps are in order
  for (size_t i = 0; i < acq_times.size(); i++) {
    auto rx = inputs_.get()[i];
    auto tx = outputs_.get()[i];
    const auto& timestamps = acq_times[i];
    // logging
    std::stringstream ss;
    for (auto ts : timestamps) ss << ts << ",";
    GXF_LOG_DEBUG("Input %ld queue: %s", i, ss.str().c_str());
    for (size_t j = timestamps.size(); j > 0; j--) {
      auto acq_time = timestamps[j - 1];
      if (acq_time < latest - threshold) {
        // drop the stale message
        GXF_LOG_DEBUG("Dropping message from input %ld at %zd", i, acq_time);
        rx->receive();
      } else if (can_send && acq_time <= latest + threshold) {
        // push the synchronized message
        GXF_LOG_DEBUG("Sending message from input %ld at %zd", i, acq_time);
        auto message = rx->receive();
        if (!message) {
          GXF_LOG_ERROR("Receiver queue corrupted, message not found");
          return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
        }
        tx->publish(message.value());
        // send only one message per output per tick
        break;
      }
    }
  }
  // if there are not enough synchronized messages, don't clear the queue, just wait for more;
  // any stale messages will be dropped on subsequent ticks

  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
