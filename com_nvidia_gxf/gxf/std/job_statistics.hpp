/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_GXF_STD_JOB_STATISTICS_SINK_HPP_
#define NVIDIA_GXF_GXF_STD_JOB_STATISTICS_SINK_HPP_

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

#include "gxf/core/component.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/gems/utils/exponential_moving_average.hpp"
#include "gxf/std/gems/utils/fast_running_median.hpp"
#include "gxf/std/ipc_server.hpp"

namespace nvidia {
namespace gxf {

/// @brief Gathers running statistics using provided clock
class JobStatistics : public Component {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  // Collects job data before running
  gxf_result_t preJob(gxf_uid_t eid);
  // Collects job data after running
  gxf_result_t postJob(gxf_uid_t eid, int64_t ticking_variation);

  // check if CodeletStats is enabled
  bool isCodeletStatistics();

  // Collects stats before running a codelet
  gxf_result_t preTick(gxf_uid_t eid, gxf_uid_t cid);
  // Collects stats after running a codelet
  gxf_result_t postTick(gxf_uid_t eid, gxf_uid_t cid);

  // Collects stats on change of an scheduling term state
  gxf_result_t postTermCheck(gxf_uid_t eid, gxf_uid_t cid, std::string next_type);
  // Collects stats on change of an entity state
  gxf_result_t onLifecycleChange(gxf_uid_t eid, std::string next_state);

  // Data structure storing state of an entity / scheduling term
  typedef struct {
    // starting timestamp of the state
    int64_t timestamp;
    // state of the entity / scheduling term
    std::string state;
  } state_record;

  // Data structure storing statistics data for one entity
  typedef struct {
    // Returns total execution time in seconds
    double getExecutionTime() const;
    // Returns total idle time in seconds
    double getIdleTime() const;

    // Running medians of the execution time
    math::FastRunningMedian<double> execution_time_median_seconds;

    // accumulated execution time in ns
    int64_t total_execution_time = 0;
    int64_t total_idle_time = 0;

    // Numbers of successful run
    int64_t tick_count = 0;

    // last seen timestamp of preJob()
    int64_t last_start_timestamp = 0;
    // last seen timestamp of postJob()
    int64_t last_stop_timestamp = 0;
    // Running median of the variation (late ticking) in periodically scheduled tasks
    math::FastRunningMedian<int64_t> late_ticking_median_ns;

    // last timestamp when the state had changed
    int64_t last_state_change_timestamp = 0;
    // Running median of state change statistics
    std::unordered_map<std::string, math::FastRunningMedian<double>> state_change_stats;
    // History of most recent event states
    std::deque<state_record> state_change_logs;
  } EntityData;

  // Data structure storing statistics data for scheduling terms of one entity
  typedef struct {
    // last scheduling condition type change timestamp
    int64_t last_change_timestamp = 0;
    // Running median of state change statistics
    std::unordered_map<std::string, math::FastRunningMedian<double>> term_change_stats;
    // History of most recent type changes
    std::deque<state_record> term_change_logs;
  }EntityTermData;

  // Data structure storing statistics data for one codelet
  typedef struct {
    // Running medians of the execution time
    math::FastRunningMedian<double> execution_time_median_seconds;

    // Numbers of successful run
    int64_t tick_count = 0;

    // last seen timestamp of preJob()
    int64_t last_start_timestamp = 0;
    // last seen timestamp of postJob()
    int64_t last_stop_timestamp = 0;

    // total codelet execution time including all ticks
    double total_execution_time = 0;
  } CodeletData;

  // Provides access to a copy of the statistics collected for all entities
  std::unordered_map<gxf_uid_t, EntityData> getallEntityData();

  // Provides access to a copy of the statistics collected for one entity
  Expected<EntityData> getEntityData(gxf_uid_t eid);

  // Provides access to a copy of the statistics collected for
  // scheduling terms of all entities
  std::unordered_map<gxf_uid_t,
                     std::unordered_map<gxf_uid_t, EntityTermData>> getallSchedulingTermData();

  // Provides access to a copy of the statistics collected for
  // scheduling terms of one entity
  Expected<std::unordered_map<gxf_uid_t,
                              EntityTermData>> getEntitySchedulingTermData(gxf_uid_t eid);

  // Provides access to a copy of the collected codelet statistics
  std::unordered_map<gxf_uid_t, std::unordered_map<gxf_uid_t, CodeletData>> getCodeletData();

 private:
  Expected<std::string> getEntityStatistics(gxf_uid_t uid);
  Expected<std::string> getCodeletStatistics(gxf_uid_t uid);
  Expected<std::string> getSchedulingEventStatistics(gxf_uid_t uid);
  Expected<std::string> getSchedulingTermStatistics(gxf_uid_t uid);
  Expected<std::string> onGetStatistics(const std::string& resource);

  // Formats the statistics and prints on the console
  Expected<void> printStatisticsOnConsole();

  // Save statistics data in a JSON file
  Expected<void> saveStatisticsInJson();

  // helper function to obtain entity and codelet name from id
  Expected<std::string> findParameterName(gxf_uid_t);

  // helper function to obtain component type name from id
  Expected<std::string> findComponentTypeName(gxf_uid_t);

  // Clock used for calculation execution timings
  Parameter<Handle<Clock>> clock_;

  // Map to store statistics for each entity
  std::unordered_map<gxf_uid_t, EntityData> entity_data_;

  // Map to store statistics for each entity
  std::unordered_map<gxf_uid_t, std::unordered_map<gxf_uid_t, EntityTermData>> entity_term_data_;

  // Mutex to prevent race conditions while accessing entity_data_
  std::shared_timed_mutex mutex_;

  // switch to enable collection of codelet performance statistics
  Parameter<bool> codelet_statistics_;

  // Length of history in terms of events count
  Parameter<uint32_t> event_history_count_;

  // Map to store codelet statistics for each entity
  mutable std::unordered_map<gxf_uid_t, std::unordered_map<gxf_uid_t, CodeletData>> codelet_data_;

  // Mutex to prevent race conditions while accessing codelet_data_
  mutable std::mutex mutex_codelet_data_;

  // JSON file path to save collected statistics
  Parameter<FilePath> json_file_name_;

  // API Server for remote access to the realtime statistic data
  Parameter<Handle<IPCServer>> server_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_GXF_STD_JOB_STATISTICS_SINK_HPP_
