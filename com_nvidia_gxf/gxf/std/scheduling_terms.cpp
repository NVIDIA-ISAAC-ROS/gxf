/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/scheduling_terms.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "gxf/std/gems/utils/time.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t PeriodicSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      recess_period_, "recess_period", "Recess Period",
      "The recess period indicates the minimum amount of time which has to pass before the entity "
      "is permitted to execute again. The period is specified as a string containing of a number "
      "and an (optional) unit. If no unit is given the value is assumed to be in nanoseconds. "
      "Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz");
  result &= registrar->parameter(
      policy_, "policy", "Policy",
      "How the scheduler handles the recess period: CatchUpMissedTicks (default), "
      "MinTimeBetweenTicks, or NoCatchUpMissedTicks",
      PeriodicSchedulingPolicy::kCatchUpMissedTicks);
  return ToResultCode(result);
}

gxf_result_t PeriodicSchedulingTerm::initialize() {
  const auto maybe = ParseRecessPeriodString(recess_period_, cid());
  if (!maybe) { return maybe.error(); }
  recess_period_ns_ = maybe.value();
  next_target_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  return GXF_SUCCESS;
}

gxf_result_t PeriodicSchedulingTerm::check_abi(int64_t timestamp, SchedulingConditionType* type,
                                               int64_t* target_timestamp) const {
  if (!next_target_) {
    *type = SchedulingConditionType::READY;
    *target_timestamp = timestamp;
    return GXF_SUCCESS;
  }

  *target_timestamp = *next_target_;
  *type = timestamp > *target_timestamp ? SchedulingConditionType::READY
                                        : SchedulingConditionType::WAIT_TIME;
  return GXF_SUCCESS;
}

gxf_result_t PeriodicSchedulingTerm::onExecute_abi(int64_t timestamp) {
  if (next_target_) {
    switch (policy_) {
      case PeriodicSchedulingPolicy::kCatchUpMissedTicks:
        next_target_ = next_target_.value() + recess_period_ns_;
        break;
      case PeriodicSchedulingPolicy::kMinTimeBetweenTicks:
        next_target_ = timestamp + recess_period_ns_;
        break;
      case PeriodicSchedulingPolicy::kNoCatchUpMissedTicks:
        next_target_ = next_target_.value() +
          ((timestamp - next_target_.value()) / recess_period_ns_ + 1) * recess_period_ns_;
        break;
    }
  } else {
    next_target_ = timestamp + recess_period_ns_;
  }
  return GXF_SUCCESS;
}

gxf_result_t TargetTimeSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(clock_, "clock", "Clock", "The clock used to define target time.");
  return ToResultCode(result);
}

gxf_result_t TargetTimeSchedulingTerm::initialize() {
  last_timestamp_ = clock_->timestamp();
  target_timestamp_ = last_timestamp_;
  return GXF_SUCCESS;
}

gxf_result_t TargetTimeSchedulingTerm::setNextTargetTime(int64_t target_timestamp) {
  // Make sure the timestamps set for execution are monotonically increasing.
  if (locked_target_timestamp_ && *locked_target_timestamp_ > target_timestamp) {
    GXF_LOG_ERROR(
        "Next target timestamp (%zd) should be on or after the current target timestamp "
        "(%zd)",
        target_timestamp, *locked_target_timestamp_);
    return GXF_FAILURE;
  }
  target_timestamp_ = target_timestamp;
  GxfEntityNotifyEventType(context(), eid(), GXF_EVENT_TIME_UPDATE);
  return GXF_SUCCESS;
}

gxf_result_t TargetTimeSchedulingTerm::check_abi(int64_t timestamp, SchedulingConditionType* type,
                                                 int64_t* target_timestamp) const {
  // Lock in the target time
  // if entity is not executed, dont invalidate the target_timestamp
  if (target_timestamp_ && !locked_target_timestamp_) {
    locked_target_timestamp_ = target_timestamp_;
    target_timestamp_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  }

  // Wait if target time is not set
  if (!locked_target_timestamp_) {
    *type = SchedulingConditionType::WAIT;
    return GXF_SUCCESS;
  }

  // Wait for the requested time target_timestamp
  *target_timestamp = *locked_target_timestamp_;
  *type = *locked_target_timestamp_ > timestamp ? SchedulingConditionType::WAIT_TIME
                                                : SchedulingConditionType::READY;
  return GXF_SUCCESS;
}

gxf_result_t TargetTimeSchedulingTerm::onExecute_abi(int64_t /* timestamp */) {
  last_timestamp_ = clock_->timestamp();
  locked_target_timestamp_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  return GXF_SUCCESS;
}

gxf_result_t CountSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(count_, "count", "Count",
                                 "The total number of time this term will permit execution.");
  return ToResultCode(result);
}

gxf_result_t CountSchedulingTerm::initialize() {
  remaining_ = count_;
  current_state_ = SchedulingConditionType::READY;
  last_run_timestamp_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t CountSchedulingTerm::check_abi(int64_t timestamp, SchedulingConditionType* type,
                                            int64_t* target_timestamp) const {
  *type = current_state_;
  *target_timestamp = last_run_timestamp_;
  return GXF_SUCCESS;
}

gxf_result_t CountSchedulingTerm::update_state_abi(int64_t /*timestamp*/) {
  if (remaining_ == 0) {
    current_state_ = SchedulingConditionType::NEVER;
  }
  return GXF_SUCCESS;
}

gxf_result_t CountSchedulingTerm::onExecute_abi(int64_t timestamp) {
  remaining_--;
  if (remaining_ == 0) {
    current_state_ = SchedulingConditionType::NEVER;
  }
  last_run_timestamp_ = timestamp;
  return GXF_SUCCESS;
}

gxf_result_t DownstreamReceptiveSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      transmitter_, "transmitter", "Transmitter",
      "The term permits execution if this transmitter can publish a message, i.e. if the receiver "
      "which is connected to this transmitter can receive messages.");
  result &= registrar->parameter(
      min_size_, "min_size", "Minimum size",
      "The term permits execution if the receiver connected to the transmitter has at least "
      "the specified number of free slots in its back buffer.", 1UL);
  return ToResultCode(result);
}

gxf_result_t DownstreamReceptiveSchedulingTerm::initialize() {
  current_state_ = SchedulingConditionType::READY;
  last_state_change_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t DownstreamReceptiveSchedulingTerm::update_state_abi(int64_t timestamp) {
  if (receivers_.empty()) {
    // Entity will never tick
    return GXF_SUCCESS;
  }

  bool is_ready = true;
  for (const Handle<Receiver> receiver : receivers_) {
    const uint64_t required = receiver->back_size() + min_size_;
    const uint64_t available = receiver->capacity() - receiver->size();
    is_ready &= required <= available;
  }

  // Check for state change
  if (is_ready && current_state_ != SchedulingConditionType::READY) {
    current_state_ = SchedulingConditionType::READY;
    last_state_change_ = timestamp;
  }

  if (!is_ready && current_state_ != SchedulingConditionType::WAIT) {
    current_state_ = SchedulingConditionType::WAIT;
    last_state_change_ = timestamp;
  }
  return GXF_SUCCESS;
}

gxf_result_t DownstreamReceptiveSchedulingTerm::check_abi(int64_t timestamp,
                                                          SchedulingConditionType* type,
                                                          int64_t* target_timestamp) const {
  if (receivers_.empty()) {
    *type = SchedulingConditionType::NEVER;
    return GXF_SUCCESS;
  }

  *type =  current_state_;
  // DownstreamReceptive ST can either be READY / WAIT state
  // and the target_timestamp is inconsequential in WAIT state
  *target_timestamp = last_state_change_;
  return GXF_SUCCESS;
}

gxf_result_t DownstreamReceptiveSchedulingTerm::onExecute_abi(int64_t timestamp) {
  // Update the state if possible. Especially needed when going from READY -> WAIT state
  return update_state_abi(timestamp);
}

gxf_result_t MessageAvailableSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver", "Queue channel",
      "The scheduling term permits execution if this channel has at least a given number of "
      "messages available.");
  result &= registrar->parameter(
      min_size_, "min_size", "Minimum message count",
      "The scheduling term permits execution if the given receiver has at least the "
      "given number of messages available.", 1UL);
  result &= registrar->parameter(
      front_stage_max_size_, "front_stage_max_size", "Maximum front stage message count",
      "If set the scheduling term will only allow execution if the number of messages in the front "
      "stage does not exceed this count. It can for example be used in combination with codelets "
      "which do not clear the front stage in every tick.",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MessageAvailableSchedulingTerm::initialize() {
  current_state_ = SchedulingConditionType::WAIT;
  last_state_change_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t MessageAvailableSchedulingTerm::update_state_abi(int64_t timestamp) {
  const bool is_ready = checkMinSize() && checkFrontStageMaxSize();
  if (is_ready && current_state_ != SchedulingConditionType::READY) {
    current_state_ =  SchedulingConditionType::READY;
    last_state_change_ = timestamp;
  }

  if (!is_ready && current_state_ != SchedulingConditionType::WAIT) {
    current_state_ =  SchedulingConditionType::WAIT;
    last_state_change_ = timestamp;
  }
  return GXF_SUCCESS;
}

gxf_result_t MessageAvailableSchedulingTerm::check_abi(int64_t timestamp,
                                                       SchedulingConditionType* type,
                                                       int64_t* target_timestamp) const {
  *type =  current_state_;
  *target_timestamp = last_state_change_;
  return GXF_SUCCESS;
}

gxf_result_t MessageAvailableSchedulingTerm::onExecute_abi(int64_t timestamp) {
  return update_state_abi(timestamp);
}

bool MessageAvailableSchedulingTerm::checkMinSize() const {
  return receiver_->back_size() + receiver_->size() >= min_size_;
}

bool MessageAvailableSchedulingTerm::checkFrontStageMaxSize() const {
  const auto maybe = front_stage_max_size_.try_get();
  return !maybe || receiver_->size() <= *maybe;
}

gxf_result_t MultiMessageAvailableSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      receivers_, "receivers", "Receivers",
      "The scheduling term permits execution if the given channels have at least a given number of "
      "messages available.");
  result &= registrar->parameter(
      min_size_, "min_size", "Minimum message count",
      "The scheduling term permits execution if all given receivers together have at least the "
      "given number of messages available", 1UL , GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      sampling_mode_, "sampling_mode", "Sampling Mode",
      "The sampling method to use when checking for messages in receiver queues. Option: SumOfAll,"
      "PerReceiver", SamplingMode::kSumOfAll);
  result &= registrar->parameter(
      min_sizes_, "min_sizes", "Minimum message counts",
      "The scheduling term permits execution if all given receivers have at least the "
      "given number of messages available in this list.", Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      min_sum_, "min_sum", "Minimum sum of message counts",
      "The scheduling term permits execution if the sum of message counts of all receivers have at "
      "least the given number of messages available.", Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MultiMessageAvailableSchedulingTerm::initialize() {
  current_state_ = SchedulingConditionType::WAIT;
  last_state_change_ = 0;

  // Check if threshold params have been set based on the sampling mode
  auto min_sizes = min_sizes_.try_get();
  switch (static_cast<SamplingMode>(sampling_mode_.get())) {
    case SamplingMode::kSumOfAll: {
      if (min_size_.try_get()) {
        GXF_LOG_WARNING("'min_size' parameter in MultiMessageAvailableSchedulingTerm is deprecated."
                         " Use 'min_sum' with SumOfAll sampling mode instead");
        min_sum_.set(min_size_.try_get().value());
      } else if (!min_sum_.try_get()) {
        GXF_LOG_ERROR("'min_sum' parameter for throttler must be set when using "
                      "'SumOfAll' throttling mode");
        return GXF_PARAMETER_NOT_INITIALIZED;
      }
      break;
    }
    case SamplingMode::kPerReceiver: {
      if (!min_sizes_.try_get()) {
        GXF_LOG_ERROR("'min_sizes' parameter for throttler must be set when using "
                      "'PerReceiver' throttling mode");
        return GXF_PARAMETER_NOT_INITIALIZED;
      }
      if (min_sizes_.try_get().value().size() != receivers_.get().size()) {
        GXF_LOG_ERROR("'min_sizes' size must be the same as 'receivers' for "
                      "'PerReceiver' throttling mode");
        return GXF_PARAMETER_OUT_OF_RANGE;
      }
      break;
    }
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  return GXF_SUCCESS;
}

gxf_result_t MultiMessageAvailableSchedulingTerm::update_state_abi(int64_t timestamp) {
  // Check if any new incoming messages are available
  bool is_ready = true;
  switch (static_cast<SamplingMode>(sampling_mode_.get())) {
    case SamplingMode::kSumOfAll: {
      size_t count = 0;
      for (const auto& rx : receivers_.get()) {
        count += rx.value()->back_size() + rx.value()->size();
      }
      if (min_sum_.try_get().value() > count) {
        is_ready = false;
      }
      break;
    }
    case SamplingMode::kPerReceiver: {
      auto receivers = receivers_.get();
      auto min_sizes = min_sizes_.try_get().value();
      for (size_t i = 0; i < receivers.size(); ++i) {
        if (min_sizes.at(i).value() > (receivers.at(i).value()->back_size() +
                                      receivers.at(i).value()->size())) {
          is_ready = false;
          break;
        }
      }
      break;
    }
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  if (is_ready) {
    last_state_change_ = current_state_ == SchedulingConditionType::READY ? \
                         last_state_change_ : timestamp;
    current_state_ = SchedulingConditionType::READY;
  } else {
    // Wait for at least one condition to be true
    last_state_change_ = current_state_ == SchedulingConditionType::WAIT ? \
                         last_state_change_ : timestamp;
    current_state_ = SchedulingConditionType::WAIT;
  }
  return GXF_SUCCESS;
}

gxf_result_t MultiMessageAvailableSchedulingTerm::check_abi(int64_t timestamp,
                                                            SchedulingConditionType* type,
                                                            int64_t* target_timestamp) const {
  *type =  current_state_;
  *target_timestamp = last_state_change_;
  return GXF_SUCCESS;
}

gxf_result_t MultiMessageAvailableSchedulingTerm::onExecute_abi(int64_t timestamp ) {
  return update_state_abi(timestamp);
}

gxf_result_t ExpiringMessageAvailableSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(max_batch_size_, "max_batch_size", "Maximum Batch Size",
                                 "The maximum number of messages to be batched together. ");
  result &= registrar->parameter(
      max_delay_ns_, "max_delay_ns", "Maximum delay in nano seconds.",
      "The maximum delay from first message to wait before submitting workload anyway.");
  result &= registrar->parameter(receiver_, "receiver", "Receiver", "Receiver to watch on.");
  result &= registrar->parameter(clock_, "clock", "Clock", "Clock to get time from.");
  return ToResultCode(result);
}

gxf_result_t ExpiringMessageAvailableSchedulingTerm::initialize() {
  return GXF_SUCCESS;
}

gxf_result_t ExpiringMessageAvailableSchedulingTerm::check_abi(int64_t timestamp,
                                                               SchedulingConditionType* type,
                                                               int64_t* target_timestamp) const {
  // Assuming sync() won't be invoked on receiver when checking terms
  const int64_t receiver_size = static_cast<int64_t>(receiver_->size() + receiver_->back_size());
  // Empty queue needs no tick
  if (receiver_size <= 0) {
    *type = SchedulingConditionType::WAIT;
    return GXF_SUCCESS;
  }
  // Having enough in queue needs tick
  if (receiver_size >= max_batch_size_) {
    *type = SchedulingConditionType::READY;
    *target_timestamp = timestamp;
    return GXF_SUCCESS;
  }
  // At least one message in queue
  int64_t oldest_message_ts = 0l;
  Expected<Entity> msg_result = Unexpected{GXF_FAILURE};
  if (0 == receiver_->size()) {
    msg_result = receiver_->peekBack();
  } else {
    msg_result = receiver_->peek();
  }
  if (!msg_result) {
    // Peek fails when there is no message in queue. Wait.
    *type = SchedulingConditionType::WAIT;
    return GXF_SUCCESS;
  }

  auto timestamp_components = msg_result->findAllHeap<Timestamp>();
  if (!timestamp_components) {
    return ToResultCode(timestamp_components);
  }
  if (0 == timestamp_components->size()) {
    // Requires Timestamp instance for message age
    GXF_LOG_ERROR("Message carries no Timestamp.");
    *type = SchedulingConditionType::READY;
    return GXF_FAILURE;
  }

  oldest_message_ts = timestamp_components->front().value()->acqtime;

  const int64_t expiring_timestamp = oldest_message_ts + max_delay_ns_;
  if (expiring_timestamp <= timestamp) {
    *type = SchedulingConditionType::READY;
    *target_timestamp = expiring_timestamp;
  } else {
    *type = SchedulingConditionType::WAIT_TIME;
    *target_timestamp = expiring_timestamp;
  }
  return GXF_SUCCESS;
}

gxf_result_t ExpiringMessageAvailableSchedulingTerm::onExecute_abi(int64_t /* timestamp */) {
  return GXF_SUCCESS;
}


gxf_result_t BooleanSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(enable_tick_, "enable_tick", "Enable Tick",
                                 "The default initial condition for enabling tick.",
                                 true, GXF_PARAMETER_FLAGS_DYNAMIC);
  return ToResultCode(result);
}


gxf_result_t BooleanSchedulingTerm::check_abi(int64_t timestamp, SchedulingConditionType* type,
                                              int64_t* target_timestamp) const {
  *type = enable_tick_.get() ? SchedulingConditionType::READY : SchedulingConditionType::NEVER;
  *target_timestamp = timestamp;
  return GXF_SUCCESS;
}

gxf_result_t BooleanSchedulingTerm::onExecute_abi(int64_t dt) {
  return GXF_SUCCESS;
}

Expected<void> BooleanSchedulingTerm::enable_tick() {
  auto retval =  enable_tick_.set(true);
  const gxf_result_t code = GxfEntityNotifyEventType(context(), eid(), GXF_EVENT_STATE_UPDATE);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Entity %ld BooleanST failed to send event notification", eid());
  }
  return retval;
}

Expected<void> BooleanSchedulingTerm::disable_tick() {
  auto retval =  enable_tick_.set(false);
  const gxf_result_t code = GxfEntityNotifyEventType(context(), eid(), GXF_EVENT_STATE_UPDATE);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Entity %ld BooleanST failed to send event notification", eid());
  }
  return retval;
}

bool BooleanSchedulingTerm::checkTickEnabled() const { return enable_tick_.get(); }

gxf_result_t BTSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(is_root_, "is_root");
  return ToResultCode(result);
}

gxf_result_t BTSchedulingTerm::initialize() {
  set_condition(is_root_.get() ? SchedulingConditionType::READY
                          : SchedulingConditionType::WAIT);
  return GXF_SUCCESS;
}

gxf_result_t BTSchedulingTerm::check_abi(int64_t timestamp,
                                         SchedulingConditionType* type,
                                         int64_t* target_timestamp) const {
  if (scheduling_condition_type_ == SchedulingConditionType::NEVER ||
      scheduling_condition_type_ == SchedulingConditionType::READY) {
    *type = scheduling_condition_type_;
    *target_timestamp = timestamp;
  } else {
    *type = SchedulingConditionType::WAIT;
  }
  return GXF_SUCCESS;
}

gxf_result_t BTSchedulingTerm::set_condition(SchedulingConditionType type) {
  scheduling_condition_type_ = type;
  return GXF_SUCCESS;
}

gxf_result_t BTSchedulingTerm::onExecute_abi(int64_t dt) { return GXF_SUCCESS; }

gxf_result_t AsynchronousSchedulingTerm::initialize() {
  return GXF_SUCCESS;
}

gxf_result_t AsynchronousSchedulingTerm::check_abi(int64_t timestamp, SchedulingConditionType* type,
                                                   int64_t* target_timestamp) const {
  std::lock_guard<std::mutex> lock(event_state_mutex_);
  if (event_state_ == AsynchronousEventState::EVENT_NEVER) {
    *type = SchedulingConditionType::NEVER;
  } else if (event_state_ == AsynchronousEventState::EVENT_WAITING) {
    *type = SchedulingConditionType::WAIT_EVENT;
  } else if (event_state_ == AsynchronousEventState::WAIT) {
    *type = SchedulingConditionType::WAIT;
  } else {
    //  EVENT_DONE / READY state so time to tick
    *type = SchedulingConditionType::READY;
    *target_timestamp = timestamp;
  }
  return GXF_SUCCESS;
}

gxf_result_t AsynchronousSchedulingTerm::onExecute_abi(int64_t dt) {
  return GXF_SUCCESS;
}

void AsynchronousSchedulingTerm::setEventState(AsynchronousEventState state) {
  std::lock_guard<std::mutex> lock(event_state_mutex_);
  event_state_ = state;
  if (event_state_ == AsynchronousEventState::EVENT_DONE) {
      GXF_LOG_DEBUG("Sending event notification for entity %ld", eid());
      GxfEntityEventNotify(context(), eid());
  }
}

AsynchronousEventState AsynchronousSchedulingTerm::getEventState() const {
  std::lock_guard<std::mutex> lock(event_state_mutex_);
  return event_state_;
}

gxf_result_t MessageAvailableFrequencyThrottler::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      execution_frequency_text_, "execution_frequency", "Execution frequency of the entity",
      "The recess period indicates the minimum amount of time which has to pass before the entity "
      "is permitted to execute again. The period is specified as a string containing of a number "
      "and an (optional) unit. If no unit is given the value is assumed to be in nanoseconds. "
      "Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz");
  result &= registrar->parameter(
      receivers_, "receivers", "Receivers",
      "The scheduling term permits execution if the given channels have at least a given number of "
      "messages available.");
  result &= registrar->parameter(
      sampling_mode_, "sampling_mode", "Sampling Mode",
      "The sampling method to use when checking for messages in receiver queues. Option: SumOfAll,"
      "PerReceiver");
  result &= registrar->parameter(
      min_sizes_, "min_sizes", "Minimum message counts",
      "The scheduling term permits execution if all given receivers have at least the "
      "given number of messages available in this list.", Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      min_sum_, "min_sum", "Minimum sum of message counts",
      "The scheduling term permits execution if the sum of message counts of all receivers have at "
      "least the given number of messages available.", Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MessageAvailableFrequencyThrottler::initialize() {
  const auto maybe_frequency = ParseRecessPeriodString(execution_frequency_text_, cid());
  if (!maybe_frequency) { return maybe_frequency.error(); }
  execution_frequency_ = maybe_frequency.value();

  // Check if threshold params have been set based on the sampling mode
  auto min_sizes = min_sizes_.try_get();
  switch (static_cast<SamplingMode>(sampling_mode_.get())) {
    case SamplingMode::kSumOfAll: {
      if (!min_sum_.try_get()) {
        GXF_LOG_ERROR("'min_sum' parameter for throttler must be set when using "
                      "'SumOfAll' throttling mode");
        return GXF_PARAMETER_NOT_INITIALIZED;
      }
      break;
    }
    case SamplingMode::kPerReceiver: {
      if (!min_sizes_.try_get()) {
        GXF_LOG_ERROR("'min_sizes' parameter for throttler must be set when using "
                      "'PerReceiver' throttling mode");
        return GXF_PARAMETER_NOT_INITIALIZED;
      }
      if (min_sizes_.try_get().value().size() != receivers_.get().size()) {
        GXF_LOG_ERROR("'min_sizes' size must be the same as 'receivers' for "
                      "'PerReceiver' throttling mode");
        return GXF_PARAMETER_OUT_OF_RANGE;
      }
      break;
    }
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  return GXF_SUCCESS;
}

gxf_result_t MessageAvailableFrequencyThrottler::update_state_abi(int64_t timestamp) {
  int64_t target_timestamp = 0;
  if (last_run_timestamp_) {
    // Tries to follow ideal rhythm
    target_timestamp = last_run_timestamp_.value() + execution_frequency_;
  } else {
    target_timestamp = execution_frequency_;
  }

  // Check if ready according to execution frequency
  if (target_timestamp <= timestamp) {
    last_state_change_ = current_state_ == SchedulingConditionType::READY ? \
                         last_state_change_ : timestamp;
    current_state_ = SchedulingConditionType::READY;
    return GXF_SUCCESS;
  }

  // Check if any new incoming messages are available
  bool is_ready = true;
  switch (static_cast<SamplingMode>(sampling_mode_.get())) {
    case SamplingMode::kSumOfAll: {
      size_t count = 0;
      for (const auto& rx : receivers_.get()) {
        count += rx.value()->back_size() + rx.value()->size();
      }
      if (min_sum_.try_get().value() > count) {
        is_ready = false;
      }
      break;
    }
    case SamplingMode::kPerReceiver: {
      auto receivers = receivers_.get();
      auto min_sizes = min_sizes_.try_get().value();
      for (size_t i = 0; i < receivers.size(); ++i) {
        if (min_sizes.at(i).value() > (receivers.at(i).value()->back_size() +
                                      receivers.at(i).value()->size())) {
          is_ready = false;
          break;
        }
      }
      break;
    }
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
  }

  if (is_ready) {
    last_state_change_ = current_state_ == SchedulingConditionType::READY ? \
                         last_state_change_ : timestamp;
    current_state_ = SchedulingConditionType::READY;
  } else {
    // Wait for at least one condition to be true
    last_state_change_ = current_state_ == SchedulingConditionType::WAIT ? \
                         last_state_change_ : timestamp;
    current_state_ = SchedulingConditionType::WAIT;
  }
  return GXF_SUCCESS;
}

gxf_result_t MessageAvailableFrequencyThrottler::check_abi(int64_t timestamp,
                                                           SchedulingConditionType* condition_type,
                                                           int64_t* target_timestamp) const {
  *condition_type =  current_state_;
  *target_timestamp = last_state_change_;
  return GXF_SUCCESS;
}

gxf_result_t MessageAvailableFrequencyThrottler::onExecute_abi(int64_t timestamp) {
  last_run_timestamp_ = timestamp;
  return update_state_abi(timestamp);
}

gxf_result_t MemoryAvailableSchedulingTerm::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(allocator_, "allocator", "Allocator",
                                 "The allocator to wait on.");
  result &= registrar->parameter(min_bytes_parameter_, "min_bytes",
                                 "Minimum bytes available",
                                 "The minimum number of bytes that must be available "
                                 "for the codelet to get scheduled. Exclusive with min_blocks.",
                                 Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(min_blocks_parameter_, "min_blocks",
                                 "Minimum blocks available",
                                 "The minimum number of blocks that must be available "
                                 "for the codelet to get scheduled. On allocators that "
                                 "do not support block allocation, this behaves the same "
                                 "as min_bytes. Exclusive with min_bytes.",
                                 Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t MemoryAvailableSchedulingTerm::initialize() {
  auto maybe_min_bytes = min_bytes_parameter_.try_get();
  auto maybe_min_blocks = min_blocks_parameter_.try_get();

  if (maybe_min_bytes && maybe_min_blocks) {
    GXF_LOG_ERROR("can only set min_bytes or min_blocks, not both");
    return GXF_PARAMETER_ALREADY_REGISTERED;
  }

  if (!maybe_min_bytes && !maybe_min_blocks) {
    GXF_LOG_ERROR("need to specify one of min_bytes or min_blocks");
    return GXF_PARAMETER_MANDATORY_NOT_SET;
  }

  if (maybe_min_blocks) {
    min_bytes_ = (allocator_->block_size() * (*maybe_min_blocks));
  } else {
    min_bytes_ = *maybe_min_bytes;
  }

  // set initial state
  current_state_ = SchedulingConditionType::WAIT;
  last_state_change_ = 0;

  return GXF_SUCCESS;
}

gxf_result_t MemoryAvailableSchedulingTerm::update_state_abi(int64_t timestamp) {
  // check whether we have enough blocks
  const bool is_ready = allocator_->is_available(min_bytes_);

  // update state if result is different
  if (is_ready && current_state_ != SchedulingConditionType::READY) {
    current_state_ = SchedulingConditionType::READY;
    last_state_change_ = timestamp;
  }

  if (!is_ready && current_state_ != SchedulingConditionType::WAIT) {
    current_state_ = SchedulingConditionType::WAIT;
    last_state_change_ = timestamp;
  }

  return GXF_SUCCESS;
}

gxf_result_t MemoryAvailableSchedulingTerm::check_abi(int64_t timestamp,
                                                      SchedulingConditionType* type,
                                                      int64_t* target_timestamp) const {
  *type = current_state_;
  *target_timestamp = last_state_change_;
  return GXF_SUCCESS;
}

gxf_result_t MemoryAvailableSchedulingTerm::onExecute_abi(int64_t timestamp) {
  return update_state_abi(timestamp);
}

}  // namespace gxf
}  // namespace nvidia
