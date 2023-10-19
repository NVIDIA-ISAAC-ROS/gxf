/*
Copyright (c) 2020,2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_GREEDY_SCHEDULER_HPP_
#define NVIDIA_GXF_STD_GREEDY_SCHEDULER_HPP_

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/fixed_vector.hpp"
#include "common/logger.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/gems/event_list/event_list.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/scheduler.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/system.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

constexpr int64_t kUsToNs = 1'000l;      // Convenient constant of 1 us = 1e3 ns

/// @brief A basic single-threaded scheduler which tests scheduling term greedily
///
/// This scheduler is great for simple use cases and predictable execution. It evaluates
/// scheduling terms greedily and may incure a large overhead of scheduling term execution. Thus it
/// may not be suitable for large applications.
///
/// The scheduler requires a Clock to keep track of time. Based on the choice of clock the scheduler
/// will execute differently. If a Realtime clock is used the scheduler will execute in realtime.
/// This means for example pausing execution, i.e. sleeping the thread, until periodic scheduling
/// terms are due again. If a ManualClock is used scheduling will happen "time-compressed". This
/// means flow of time is altered to execute codelets immediately after each other.
class GreedyScheduler : public Scheduler {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  gxf_result_t prepare_abi(EntityExecutor* executor) override;
  gxf_result_t schedule_abi(gxf_uid_t eid) override;
  gxf_result_t unschedule_abi(gxf_uid_t eid) override;
  gxf_result_t runAsync_abi() override;
  gxf_result_t stop_abi() override;
  gxf_result_t wait_abi() override;
  gxf_result_t event_notify_abi(gxf_uid_t eid) override;

 private:
  Parameter<Handle<Clock>> clock_;
  Parameter<bool> realtime_;
  Parameter<int64_t> max_duration_ms_;
  Parameter<bool> stop_on_deadlock_;
  Parameter<double> check_recession_period_ms_;
  Parameter<int64_t> stop_on_deadlock_timeout_;
  Parameter<int64_t> check_recession_period_us_;

  EntityExecutor* executor_ = nullptr;

  std::atomic_bool stopping_{true};
  std::atomic<gxf_result_t> thread_error_code_;
  std::unique_ptr<std::thread> thread_ = nullptr;

  // Used temporarily until realtime flag is removed.
  Entity clock_entity_;

  // Used for keeping track of async events
  uint64_t count_wait_event_{0};
  std::mutex event_mutex_;
  std::unique_ptr<EventList<gxf_uid_t>> event_notified_;
  std::unique_ptr<EventList<gxf_uid_t>> event_waiting_;
  std::condition_variable event_notification_cv_;

  // Used for keeping track of graph entities
  FixedVector<gxf_uid_t> active_entities_;
  FixedVector<gxf_uid_t> new_entities_;
  std::unique_ptr<EventList<gxf_uid_t>> unschedule_entities_;
  std::mutex entity_mutex_;
  // Dedicated mutex for each entity
  std::unordered_map<gxf_uid_t, std::unique_ptr<std::mutex>> entity_mutex_map_;

  // latest timestamp from last should_stop == false
  int64_t last_no_stop_ts_ = 0;
  // maintain last no stop timestamp, and check if need to update should_stop
  gxf_result_t stop_on_deadlock_timeout(const int64_t timeout, const int64_t now,
    bool& should_stop);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_GREEDY_SCHEDULER_HPP_
