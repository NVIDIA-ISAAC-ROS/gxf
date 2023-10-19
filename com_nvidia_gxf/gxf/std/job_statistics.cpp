/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/job_statistics.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/assert.hpp"
#include "gxf/std/gems/utils/time.hpp"
#include "nlohmann-json/json.hpp"

namespace nvidia {
namespace gxf {
namespace {
// clang-format off
constexpr char kConsoleStatisticsHeader[] =
    "|==================================================================================================================================================================|\n"  // NOLINT
    "|                                           Job Statistics Report (regular)                                                                                        |\n"  // NOLINT
    "|==================================================================================================================================================================|\n"  // NOLINT
    "| Name                                               |   Count | Time (Median - 90% - Max) [ms] | Load (%) | Exec(ms) | Variation (Median - 90% - Max) [ns]        |\n"  // NOLINT
    "|------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n"; // NOLINT

constexpr char kConsoleStatisticsItemLine[] =
    "| %50.50s | %7zd | %8.2f | %8.2f | %8.2f | %6.1f %% |   %6.1f |   %10ld |   %10ld |   %10ld |\n";  // NOLINT

constexpr char kConsoleStatisticsFooter[] =
    "|==================================================================================================================================================================|";   // NOLINT

constexpr char kConsoleCodeletStatisticsHeader[] =
    "|==================================================================================================================================================================|\n"  // NOLINT
    "|                                           Codelet Statistics Report (regular)                                                                                    |\n"  // NOLINT
    "|==================================================================================================================================================================|\n"  // NOLINT
    "| Entity Name              | Codelet Name            |   Count | Time (Mean - 90% - Max) [ms]   | Frequency (ticks/ms)                                             |\n"  // NOLINT
    "|------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n"; // NOLINT

constexpr char kConsoleCodeletStatisticsItemLine[] =
    "| %23.23s  | %23.23s | %7zd | %8.5f | %8.5f | %8.5f |   %6.1f                                                         |\n";  // NOLINT

constexpr char kConsoleEntityStatisticsHeader[] =
    "|==================================================================================================================================================================|\n"  // NOLINT
    "|                                           Entity Statistics Report (regular)                                                                                     |\n"  // NOLINT
    "|==================================================================================================================================================================|\n"  // NOLINT
    "| Entity Name             | Entity State             |   Count | Time (Median - 90% - Max) [ms]                                                                    |\n"  // NOLINT
    "|------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n"; // NOLINT

constexpr char kConsoleEntityStatisticsItemLine[] =
    "| %23.23s  | %24.24s | %7zd | %8.5f | %8.5f | %8.5f                                                                   |\n";  // NOLINT

constexpr char kConsoleEntitySeperator[] =
    "|------------------------------------------------------------------------------------------------------------------------------------------------------------------|";   // NOLINT

// clang-format on

// Returns a new string containing the last count characters of a string
inline std::string TakeLast(const std::string& str, size_t count) {
  if (str.length() <= count) {
    return str;
  } else {
    if (count < 2) {
      return str.substr(str.length() - count, count);
    } else {
      return ".." + str.substr(str.length() - count + 2, count - 2);
    }
  }
}

}  // namespace

gxf_result_t JobStatistics::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(clock_, "clock", "Clock",
                                 "The clock component instance to retrieve time from.");
  result &=
      registrar->parameter(codelet_statistics_, "codelet_statistics", "Codelet Statistics",
                           "Parameter to enable/disable statistics collection for Codelets", false);

  result &= registrar->parameter(json_file_name_, "json_file_path", "JSON File Path",
                                 "JSON file path to save statistics output",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(server_, "server", "API server",
                                 "API Server for remote access to the realtime statistic data",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(event_history_count_, "event_history_count", "History event count",
                                "Count of the number of events to be maintained in history per "
                                "entity / scheduling term", 100u);
  return ToResultCode(result);
}

gxf_result_t JobStatistics::preJob(gxf_uid_t eid) {
  auto it = entity_data_.find(eid);
  if (it == entity_data_.end()) {
    std::lock_guard<std::shared_timed_mutex> lock(mutex_);
    entity_data_[eid] = EntityData{};
    entity_term_data_[eid] = std::unordered_map<gxf_uid_t, EntityTermData>{};
    it = entity_data_.find(eid);
  }

  const int64_t now = clock_->timestamp();
  if (now < it->second.last_stop_timestamp) {
    GXF_LOG_ERROR("Invalid timestamp for last stop %ld now %ld", it->second.last_stop_timestamp,
                  now);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  it->second.last_start_timestamp = now;
  return GXF_SUCCESS;
}

gxf_result_t JobStatistics::postJob(gxf_uid_t eid, int64_t ticking_variation) {
  const int64_t now = clock_->timestamp();

  auto it = entity_data_.find(eid);
  if (it == entity_data_.end()) {
    GXF_LOG_ERROR("No previous record for eid %lu ", eid);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }
  if (now < it->second.last_start_timestamp) {
    GXF_LOG_ERROR("Invalid timestamp for last start %ld now %ld",
                  it->second.last_start_timestamp, now);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  it->second.tick_count++;
  if (it->second.last_stop_timestamp > 0) {
    it->second.total_idle_time += it->second.last_start_timestamp - it->second.last_stop_timestamp;
  }

  it->second.last_stop_timestamp = now;

  const int64_t dt = now - it->second.last_start_timestamp;
  const double dt_seconds = TimestampToTime(dt);
  it->second.total_execution_time += dt;
  it->second.execution_time_median_seconds.add(dt_seconds);
  it->second.late_ticking_median_ns.add(ticking_variation);

  return GXF_SUCCESS;
}

double JobStatistics::EntityData::getExecutionTime() const {
  return TimestampToTime(total_execution_time);
}

double JobStatistics::EntityData::getIdleTime() const {
  return TimestampToTime(total_idle_time);
}

gxf_result_t JobStatistics::initialize() {
  auto maybe_server = server_.try_get();
  if (maybe_server) {
    auto server_handle = maybe_server.value();
    IPCServer::Service service = {
      "stat",
      IPCServer::kQuery,
      {.query = std::bind(&JobStatistics::onGetStatistics, this, std::placeholders::_1)}
    };
    server_handle->registerService(service);
  }
  return GXF_SUCCESS;
}

Expected<void> JobStatistics::printStatisticsOnConsole() {
  // Print JobStatistics in the form of a table
  std::printf("%s", kConsoleStatisticsHeader);
  for (const auto& pair : entity_data_) {
    auto this_name = findParameterName(pair.first);
    if (!this_name) {
      GXF_LOG_ERROR("Error retrieving entity name for eid %lu", pair.first);
      return Unexpected{GXF_FAILURE};
    }
    const EntityData& stats = pair.second;
    double load = 100.0f;
    if (stats.getIdleTime() > 0.0f) {
      load = 100.0 * stats.getExecutionTime() / (stats.getExecutionTime() + stats.getIdleTime());
    }
    std::printf(kConsoleStatisticsItemLine, TakeLast(this_name.value(), 50).c_str(),
                stats.tick_count, stats.execution_time_median_seconds.median() * 1000,
                stats.execution_time_median_seconds.percentile(0.9) * 1000,
                std::max(stats.execution_time_median_seconds.max() * 1000, 0.0), load,
                stats.getExecutionTime() * 1000, stats.late_ticking_median_ns.median(),
                stats.late_ticking_median_ns.percentile(0.9),
                std::max(stats.late_ticking_median_ns.max(), 0L));
  }
  std::printf("%s\n", kConsoleStatisticsFooter);

  // Print CodeletStatistics in the form of a table
  if (isCodeletStatistics()) {
    std::printf("\n\n");
    std::printf("%s", kConsoleCodeletStatisticsHeader);
    for (const auto& pair : codelet_data_) {
      const gxf_uid_t eid = pair.first;
      auto this_entity_name = findParameterName(eid);
      if (!this_entity_name) {
        GXF_LOG_ERROR("Error retrieving entity name for eid %lu", eid);
        return Unexpected{GXF_FAILURE};
      }
      for (const auto& codelet_pair : codelet_data_[eid]) {
        auto this_codelet_name = findParameterName(codelet_pair.first);
        if (!this_codelet_name) {
          GXF_LOG_ERROR("Error retrieving entity name for cid %lu", codelet_pair.first);
          return Unexpected{GXF_FAILURE};
        }
        const CodeletData& stats = codelet_pair.second;
        std::printf(kConsoleCodeletStatisticsItemLine,
                    TakeLast(this_entity_name.value(), 23).c_str(),
                    TakeLast(this_codelet_name.value(), 23).c_str(), stats.tick_count,
                    (stats.total_execution_time / stats.tick_count) / 1000000,
                    stats.execution_time_median_seconds.percentile(0.9) * 1000,
                    stats.execution_time_median_seconds.max() * 1000,
                    stats.tick_count / (stats.total_execution_time / 1000000));
      }
    }
    std::printf("%s\n", kConsoleStatisticsFooter);
  }

  // Print entity state statistics
  std::printf("%s", kConsoleEntityStatisticsHeader);
  for (const auto& pair : entity_data_) {
    auto this_name = findParameterName(pair.first);
    if (!this_name) {
      GXF_LOG_ERROR("Error retrieving entity name for eid %lu", pair.first);
      return Unexpected{GXF_FAILURE};
    }

    const EntityData& stats = pair.second;
    for (const auto& state_pair : stats.state_change_stats) {
      const auto state_name = state_pair.first;
      const auto state_stat = state_pair.second;
      std::printf(kConsoleEntityStatisticsItemLine, TakeLast(this_name.value(), 50).c_str(),
                  state_name.c_str(), state_stat.count(), state_stat.median() * 1000.0,
                  state_stat.percentile(0.9) * 1000.0, std::max(state_stat.max() * 1000.0, 0.0));
    }
    std::printf("%s\n", kConsoleEntitySeperator);
  }
  std::printf("%s\n", kConsoleStatisticsFooter);
  return Success;
}

gxf_result_t JobStatistics::deinitialize() {
  auto result = printStatisticsOnConsole();
  if (!result) {
    return ToResultCode(result);
  }

  result = saveStatisticsInJson();
  if (!result) {
    return ToResultCode(result);
  }

  // #define GXF_ENABLE_DEBUG_INFO
  #ifdef GXF_ENABLE_DEBUG_INFO
  for (auto e_pair : entity_term_data_) {
    auto e_data = e_pair.second;
    GXF_LOG_INFO("Entity name %s", findParameterName(e_pair.first).value().c_str());
    for (auto c_pair : e_data) {
      auto c_data = c_pair.second;
      auto c_logs = c_data.term_change_logs;
      auto c_stats = c_data.term_change_stats;

      GXF_LOG_INFO("    Component name %s", findParameterName(c_pair.first).value().c_str());
      for (auto rec : c_stats) {
        GXF_LOG_INFO("         type: [%s] median: [%f] max: [%f] count: [%d] ", rec.first.c_str(),
                     rec.second.median() * 1000, rec.second.percentile(0.9)*1000.0,
                     rec.second.count());
      }
      // GXF_LOG_INFO("   Logs ----------------------------------------------");
      // for (auto log : c_logs) {
      //   GXF_LOG_INFO("         State: [%s] Ts: [%ld]", log.state.c_str(), log.timestamp);
      // }
    }
  }
  #endif
  return GXF_SUCCESS;
}

std::unordered_map<gxf_uid_t, JobStatistics::EntityData> JobStatistics::getallEntityData() {
  // Return a copy rather than a reference to prevent the reference from being accessed
  // while the data is being modified
  std::lock_guard<std::shared_timed_mutex> lock(mutex_);
  return entity_data_;
}

Expected<JobStatistics::EntityData> JobStatistics::getEntityData(gxf_uid_t eid) {
  std::lock_guard<std::shared_timed_mutex> lock(mutex_);
  auto it = entity_data_.find(eid);
  if (it == entity_data_.end()) {
    auto entity_name = findParameterName(eid);
    GXF_LOG_ERROR("Statistics not found for entity %s", entity_name.value().c_str());
    return Unexpected{GXF_ENTITY_NOT_FOUND};
  }
  return it->second;
}

std::unordered_map<gxf_uid_t, std::unordered_map<gxf_uid_t, JobStatistics::EntityTermData>>
JobStatistics::getallSchedulingTermData() {
  std::lock_guard<std::shared_timed_mutex> lock(mutex_);
  return entity_term_data_;
}

Expected<std::unordered_map<gxf_uid_t, JobStatistics::EntityTermData>>
  JobStatistics::getEntitySchedulingTermData(gxf_uid_t eid) {
    std::lock_guard<std::shared_timed_mutex> lock(mutex_);
    auto it = entity_term_data_.find(eid);
    if (it == entity_term_data_.end()) {
      auto entity_name = findParameterName(eid);
      GXF_LOG_ERROR("Statistics not found for entity %s", entity_name.value().c_str());
      return Unexpected{GXF_ENTITY_NOT_FOUND};
    }
    return it->second;
}

gxf_result_t JobStatistics::preTick(gxf_uid_t eid, gxf_uid_t cid) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (codelet_data_.find(eid) == codelet_data_.end()) {
    std::lock_guard<std::mutex> lock(mutex_codelet_data_);
    codelet_data_[eid] = std::unordered_map<gxf_uid_t, CodeletData>();
  }

  if (codelet_data_[eid].find(cid) == codelet_data_[eid].end()) {
    // no sync mechanism has been used inside this block since
    // we dont run multiple threads per entity
    codelet_data_[eid][cid] = CodeletData();
  }

  auto it = codelet_data_[eid].find(cid);
  const int64_t now = clock_->timestamp();
  if (now < it->second.last_stop_timestamp) {
    GXF_LOG_ERROR("Invalid timestamp for last stop %ld now %ld", it->second.last_stop_timestamp,
                  now);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  it->second.last_start_timestamp = now;

  return GXF_SUCCESS;
}

gxf_result_t JobStatistics::postTick(gxf_uid_t eid, gxf_uid_t cid) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const int64_t now = clock_->timestamp();
  if (codelet_data_.find(eid) == codelet_data_.end()) {
    GXF_LOG_ERROR("No previous record for eid %lu ", eid);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  auto it = codelet_data_[eid].find(cid);
  if (it == codelet_data_[eid].end()) {
    GXF_LOG_ERROR("No previous record for eid %lu cid %lu", eid, cid);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  if (now < it->second.last_start_timestamp) {
    GXF_LOG_ERROR("Invalid timestamp for last start %ld now %ld", it->second.last_start_timestamp,
                  now);
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  it->second.tick_count++;
  it->second.last_stop_timestamp = now;

  const int64_t dt = now - it->second.last_start_timestamp;
  const double dt_seconds = TimestampToTime(dt);
  it->second.total_execution_time += dt;
  it->second.execution_time_median_seconds.add(dt_seconds);

  return GXF_SUCCESS;
}

gxf_result_t JobStatistics::onLifecycleChange(gxf_uid_t eid, std::string next_state) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  auto it = entity_data_.find(eid);
  if (it == entity_data_.end()) {
    auto entity_name = findParameterName(eid);
    return GXF_ENTITY_NOT_FOUND;
  }

  auto& entity_data = it->second;
  const int64_t now = clock_->timestamp();
  if (now < entity_data.last_state_change_timestamp) {
    auto entity_name = findParameterName(eid);
    GXF_LOG_ERROR("Invalid timestamp for last state change %ld now %ld for entity %s",
                  entity_data.last_state_change_timestamp, now, entity_name.value().c_str());
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  // Update entity stats and entity logs
  const int64_t dt = now - entity_data.last_state_change_timestamp;
  const double dt_seconds = TimestampToTime(dt);

  // If a previous state exists, update the state duration
  // else update the logs, duration will be added in the next iteration
  if (!entity_data.state_change_logs.empty()) {
    std::string current_state = entity_data.state_change_logs.front().state;
    // Create a new median calculator if not present already
    if (entity_data.state_change_stats.find(current_state) ==
        entity_data.state_change_stats.end()) {
      entity_data.state_change_stats[current_state] = math::FastRunningMedian<double>{};
    }
    entity_data.state_change_stats.at(current_state).add(dt_seconds);
  }

  entity_data.last_state_change_timestamp = now;
  entity_data.state_change_logs.push_front(state_record{now, next_state});

  // resize the event history length
  const auto count = event_history_count_.get();
  if (entity_data.state_change_logs.size() > count) { entity_data.state_change_logs.resize(count); }

  return GXF_SUCCESS;
}

gxf_result_t JobStatistics::postTermCheck(gxf_uid_t eid, gxf_uid_t cid, std::string next_type) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  auto it = entity_term_data_.find(eid);
  if (it == entity_term_data_.end()) {
    return GXF_ENTITY_NOT_FOUND;
  }

  // All entity scheduling terms
  auto& terms_data = it->second;
  auto ij = terms_data.find(cid);
  if (ij == terms_data.end()) {
    terms_data[cid] = EntityTermData{};
    ij = terms_data.find(cid);
  }

  // Current entity scheduling term
  auto& term_data = ij->second;
  if (!term_data.term_change_logs.empty() &&
     (term_data.term_change_logs.front().state == next_type)) {
      // No change in scheduling term state, dont update any stats for now.
      return GXF_SUCCESS;
     }

  const int64_t now = clock_->timestamp();
  if (now < term_data.last_change_timestamp) {
    auto entity_name = findParameterName(eid);
    GXF_LOG_ERROR("Invalid timestamp for last condition type change %ld now %ld for entity %s",
                  term_data.last_change_timestamp, now, entity_name.value().c_str());
    return GXF_INVALID_EXECUTION_SEQUENCE;
  }

  const int64_t dt = now - term_data.last_change_timestamp;
  const double dt_seconds = TimestampToTime(dt);

  // If a previous state exists, update the state duration
  // else update the logs, duration will be added in the next iteration
  if (!term_data.term_change_logs.empty()) {
    std::string current_state = term_data.term_change_logs.front().state;
    // Create a new median calculator if not present already
    if (term_data.term_change_stats.find(current_state) == term_data.term_change_stats.end()) {
      term_data.term_change_stats[current_state] = math::FastRunningMedian<double>{};
    }
    term_data.term_change_stats.at(current_state).add(dt_seconds);
  }

  term_data.last_change_timestamp = now;
  term_data.term_change_logs.push_front(state_record{now, next_type});

  // resize the event history length
  const auto count = event_history_count_.get();
  if (term_data.term_change_logs.size() > count) { term_data.term_change_logs.resize(count); }

  return GXF_SUCCESS;
}

bool JobStatistics::isCodeletStatistics() {
  return codelet_statistics_.get();
}

std::unordered_map<gxf_uid_t, std::unordered_map<gxf_uid_t, JobStatistics::CodeletData>>
JobStatistics::getCodeletData() {
  std::lock_guard<std::mutex> lock(mutex_codelet_data_);
  return codelet_data_;
}

Expected<std::string> JobStatistics::findParameterName(gxf_uid_t id) {
  const char* this_name = nullptr;
  std::string backup_name = std::to_string(id);
  const gxf_result_t result =
      GxfParameterGetStr(context_, id, kInternalNameParameterKey, &this_name);
  if (result != GXF_SUCCESS || this_name[0] == '\0') {
    this_name = backup_name.c_str();
  }
  return this_name;
}

Expected<std::string> JobStatistics::findCodeletTypeName(gxf_uid_t id) {
  gxf_tid_t tid = GxfTidNull();
  auto code = GxfComponentType(context_, id, &tid);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component type");
    return Unexpected{GXF_FAILURE};
  }
  const char* component_type_name = nullptr;
  code = GxfComponentTypeName(context_, tid, &component_type_name);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component type name");
    return Unexpected{GXF_FAILURE};
  }

  return component_type_name;
}

Expected<void> JobStatistics::saveStatisticsInJson() {
  auto file_name = json_file_name_.try_get();
  if (!file_name) {
    // Check if JSON file path is provided
    return Success;
  }

  // Lambda Expression to roundup decimal digits
  auto DecimalPrecision5 = [](double number) {
    return static_cast<int>(((number + 0.000005) * 100000)) / 100000.0;
  };

  nlohmann::json json_object;
  {  // add job_statistics data
    std::vector<nlohmann::json> items;
    for (const auto& entity_iterator : entity_data_) {
      double load = 100.0f;
      if (entity_iterator.second.getIdleTime() > 0.0f) {
        load = 100.0 * entity_iterator.second.getExecutionTime() /
               (entity_iterator.second.getExecutionTime() + entity_iterator.second.getIdleTime());
      }
      auto entity_name = findParameterName(entity_iterator.first);
      if (!entity_name) {
        GXF_LOG_ERROR("Error retrieving entity name for eid %lu", entity_iterator.first);
        return Unexpected{GXF_FAILURE};
      }
      items.push_back({
          {"name", entity_name.value()},
          {"count", entity_iterator.second.tick_count},
          {"execution_time_median_ms",
           DecimalPrecision5(entity_iterator.second.execution_time_median_seconds.median())},
          {"execution_time_90_ms",
           DecimalPrecision5(entity_iterator.second.execution_time_median_seconds.percentile(0.9) *
                             1000)},
          {"execution_time_max_ms",
           DecimalPrecision5(entity_iterator.second.execution_time_median_seconds.max() * 1000)},
          {"load_percentage", DecimalPrecision5(load)},
          {"execution_time_ms",
           DecimalPrecision5(entity_iterator.second.getExecutionTime() * 1000)},
          {"variation_median_ns",
           DecimalPrecision5(entity_iterator.second.late_ticking_median_ns.median())},
          {"variation_90_ns",
           DecimalPrecision5(entity_iterator.second.late_ticking_median_ns.percentile(0.9) * 1000)},
          {"variation_max_ns",
           DecimalPrecision5(entity_iterator.second.late_ticking_median_ns.max() * 1000)},
      });
    }
    json_object["entities"] = items;
  }

  {  // add codelet_statistics data
    std::vector<nlohmann::json> entity_json;
    for (const auto& entity_iterator : codelet_data_) {
      auto entity_name = findParameterName(entity_iterator.first);
      if (!entity_name) {
        GXF_LOG_ERROR("Error retrieving entity name for eid %lu", entity_iterator.first);
        return Unexpected{GXF_FAILURE};
      }
      std::vector<nlohmann::json> codelet_json;
      for (const auto& codelet_iterator : entity_iterator.second) {
        auto codelet_name = findParameterName(codelet_iterator.first);
        if (!codelet_name) {
          GXF_LOG_ERROR("Error retrieving codelet name for cid %lu", codelet_iterator.first);
          return Unexpected{GXF_FAILURE};
        }
        codelet_json.push_back(
            {{"codelet_name", codelet_name.value()},
             {"count", codelet_iterator.second.tick_count},
             {"execution_time_mean_ms",
              DecimalPrecision5((codelet_iterator.second.total_execution_time /
                                 codelet_iterator.second.tick_count) /
                                1000000)},
             {"execution_time_90_ms",
              DecimalPrecision5(
                  codelet_iterator.second.execution_time_median_seconds.percentile(0.9) * 1000)},
             {"execution_time_max_ms",
              DecimalPrecision5(codelet_iterator.second.execution_time_median_seconds.max() *
                                1000)},
             {"tick_frequency_per_ms",
              DecimalPrecision5(codelet_iterator.second.tick_count /
                                (codelet_iterator.second.total_execution_time / 1000000))}});
      }
      nlohmann::json entity_item;
      entity_item["name"] = entity_name.value();
      entity_item["entity"] = codelet_json;
      entity_json.push_back(entity_item);
    }
    json_object["codelets"] = entity_json;
  }

  std::ofstream output_file(file_name.value());
  if (output_file.is_open()) {
    output_file << std::setw(4) << json_object << std::endl;
    output_file.close();
  } else {
    GXF_LOG_ERROR("Error while opening file for writing:%s\n", file_name.value().c_str());
    return Unexpected{GXF_FAILURE};
  }

  GXF_LOG_INFO("Performance Statistics has been saved in the following JSON file: %s\n",
               file_name.value().c_str());
  return Success;
}

Expected<std::string> JobStatistics::getEntityStatistics(gxf_uid_t uid) {
  GXF_LOG_INFO("JobStatistics::getCodeletStatistics Enters");
  // Lambda Expression to roundup decimal digits
  auto DecimalPrecision5 = [](double number) {
    return static_cast<int>(((number + 0.000005) * 100000)) / 100000.0;
  };

  nlohmann::json json_array = nlohmann::json::array();
  auto entity_data = getallEntityData();
  for (const auto& entity_iterator : entity_data) {
    if (uid != kUnspecifiedUid && uid != entity_iterator.first) {
      continue;
    }
    auto entity_name = findParameterName(entity_iterator.first);
    if (!entity_name) {
      GXF_LOG_ERROR("Error retrieving entity name for eid %lu", entity_iterator.first);
      return Unexpected{GXF_FAILURE};
    }
    double load = 100.0f;
    if (entity_iterator.second.getIdleTime() > 0.0f) {
      load = 100.0 * entity_iterator.second.getExecutionTime() /
              (entity_iterator.second.getExecutionTime() + entity_iterator.second.getIdleTime());
    }
    gxf_entity_status_t entity_status;
    const unsigned int STATUS_MAX = GXF_ENTITY_MAX;
    const std::string status_map[STATUS_MAX] = {
      "Not Started",
      "Start Pending",
      "Started",
      "Tick Pending",
      "Ticking",
      "Idle",
      "Stop Pending"
    };
    if ((GxfEntityGetStatus(context_, entity_iterator.first, &entity_status) != GXF_SUCCESS) ||
        (entity_status >= STATUS_MAX)
    ) {
      GXF_LOG_ERROR("Error retrieving entity status for eid %lu", entity_iterator.first);
      return Unexpected{GXF_FAILURE};
    }
    json_array.push_back({
      {"name", entity_name.value()},
      {"uid", entity_iterator.first},
      {"status", status_map[entity_status]},
      {"count", entity_iterator.second.tick_count},
      {"execution_time_median_ms",
        DecimalPrecision5(entity_iterator.second.execution_time_median_seconds.median() * 1000)},
      {"execution_time_90_ms",
        DecimalPrecision5(entity_iterator.second.execution_time_median_seconds.percentile(0.9) *
                          1000)},
      {"execution_time_max_ms",
        DecimalPrecision5(entity_iterator.second.execution_time_median_seconds.max() * 1000)},
      {"load_percentage", DecimalPrecision5(load)},
      {"execution_time_total_ms",
        DecimalPrecision5(entity_iterator.second.getExecutionTime() * 1000)},
      {"variation_median_ms",
        DecimalPrecision5(entity_iterator.second.late_ticking_median_ns.median()/1000000)},
      {"variation_90_ms",
        DecimalPrecision5(entity_iterator.second.late_ticking_median_ns.percentile(0.9)/1000000)},
      {"variation_max_ms",
        DecimalPrecision5(entity_iterator.second.late_ticking_median_ns.max()/1000000)},
    });
  }
  GXF_LOG_INFO("JobStatistics::getCodeletStatistics Leaves");
  return json_array.dump();
}

Expected<std::string> JobStatistics::getCodeletStatistics(gxf_uid_t uid) {
  GXF_LOG_INFO("JobStatistics::getCodeletStatistics Enters");
  // Lambda Expression to roundup decimal digits
  auto DecimalPrecision5 = [](double number) {
    return static_cast<int>(((number + 0.000005) * 100000)) / 100000.0;
  };

  nlohmann::json json_array = nlohmann::json::array();
  auto codelet_data = getCodeletData();
  bool filter_eid = false;
  if (uid != kUnspecifiedUid && codelet_data.find(uid) != codelet_data.end()) {
    filter_eid = true;
  }
  for (const auto entity_iterator : codelet_data) {
    if (filter_eid && entity_iterator.first != uid) {
      continue;
    }
    auto entity_name = findParameterName(entity_iterator.first);
    if (!entity_name) {
      GXF_LOG_ERROR("Error retrieving entity name for eid %lu", entity_iterator.first);
      return Unexpected{GXF_FAILURE};
    }
    for (const auto& codelet_iterator : entity_iterator.second) {
      if (uid != kUnspecifiedUid && !filter_eid && uid != codelet_iterator.first) {
        continue;
      }
        auto codelet_name = findParameterName(codelet_iterator.first);
        if (!codelet_name) {
          GXF_LOG_ERROR("Error retrieving codelet name for cid %lu", codelet_iterator.first);
          return Unexpected{GXF_FAILURE};
        }
        auto codelet_type_name = findCodeletTypeName(codelet_iterator.first);
        if (!codelet_type_name) {
          GXF_LOG_ERROR("Error retrieving codelet typename for cid %lu", codelet_iterator.first);
          return Unexpected{GXF_FAILURE};
        }
        json_array.push_back({
          {"name", codelet_name.value()},
          {"uid", codelet_iterator.first},
          {"type", codelet_type_name.value()},
          {"entity", entity_name.value()},
          {"ticks", codelet_iterator.second.tick_count},
          {"execution_time_mean_ms",
          DecimalPrecision5((codelet_iterator.second.total_execution_time /
                              codelet_iterator.second.tick_count) /
                            1000000)},
          {"execution_time_90_ms",
          DecimalPrecision5(
              codelet_iterator.second.execution_time_median_seconds.percentile(0.9) * 1000)},
          {"execution_time_max_ms",
          DecimalPrecision5(codelet_iterator.second.execution_time_median_seconds.max() *
                            1000)},
          {"tick_frequency_per_ms",
          DecimalPrecision5(codelet_iterator.second.tick_count /
                            (codelet_iterator.second.total_execution_time / 1000000))}
        });
    }
  }
  GXF_LOG_INFO("JobStatistics::getCodeletStatistics Leaves");
  return json_array.dump();
}

Expected<std::string> JobStatistics::getSchedulingEventStatistics(gxf_uid_t uid) {
  GXF_LOG_INFO("JobStatistics::getSchedulingEventStatistics Enters");
  nlohmann::json json_array  = nlohmann::json::array();
  auto all_term_data = getallSchedulingTermData();
  for (const auto entity_iterator : all_term_data) {
    if (uid != kUnspecifiedUid && uid != entity_iterator.first) {
      continue;
    }
    auto entity_name = findParameterName(entity_iterator.first);
    if (!entity_name) {
      GXF_LOG_ERROR("Error retrieving entity name for eid %lu", entity_iterator.first);
      return Unexpected{GXF_FAILURE};
    }
    auto term_data = entity_iterator.second;
    nlohmann::json term_array = nlohmann::json::array();
    for (const auto term_iterator : term_data) {
      auto term_name = findParameterName(term_iterator.first);
      if (!term_name) {
        GXF_LOG_ERROR("Error retrieving scheduling term name for cid %lu", term_iterator.first);
        return Unexpected{GXF_FAILURE};
      }
      auto term_type_name = findCodeletTypeName(term_iterator.first);
      if (!term_type_name) {
        GXF_LOG_ERROR("Error retrieving scheduling term typename for cid %lu", term_iterator.first);
        return Unexpected{GXF_FAILURE};
      }
      nlohmann::json log_array = nlohmann::json::array();
      for (const auto log_iterator : term_iterator.second.term_change_logs) {
        log_array.push_back({
          {"timestamp", log_iterator.timestamp},
          {"state", log_iterator.state}
        });
      }
      term_array.push_back({
        {"name", term_name.value()},
        {"uid", term_iterator.first},
        {"type", term_type_name.value()},
        {"data", log_array}
      });
    }

    json_array.push_back({
      {"name", entity_name.value()},
      {"uid", entity_iterator.first},
      {"data", term_array}
    });
  }
  GXF_LOG_INFO("JobStatistics::getSchedulingEventStatistics Leaves");
  return json_array.dump();
}

Expected<std::string> JobStatistics::getSchedulingTermStatistics(gxf_uid_t uid) {
  GXF_LOG_INFO("JobStatistics::getSchedulingTermStatistics Enters");
  nlohmann::json json_array  = nlohmann::json::array();
  auto all_term_data = getallSchedulingTermData();
  for (const auto entity_iterator : all_term_data) {
    if (uid != kUnspecifiedUid && uid != entity_iterator.first) {
      continue;
    }
    auto entity_name = findParameterName(entity_iterator.first);
    if (!entity_name) {
      GXF_LOG_ERROR("Error retrieving entity name for eid %lu", entity_iterator.first);
      return Unexpected{GXF_FAILURE};
    }
    auto term_data = entity_iterator.second;
    nlohmann::json term_array = nlohmann::json::array();
    for (const auto term_iterator : term_data) {
      auto term_name = findParameterName(term_iterator.first);
      if (!term_name) {
        GXF_LOG_ERROR("Error retrieving scheduling term name for cid %lu", term_iterator.first);
        return Unexpected{GXF_FAILURE};
      }
      auto term_type_name = findCodeletTypeName(term_iterator.first);
      if (!term_type_name) {
        GXF_LOG_ERROR("Error retrieving scheduling term typename for cid %lu", term_iterator.first);
        return Unexpected{GXF_FAILURE};
      }
      nlohmann::json stats_array = nlohmann::json::array();
      for (const auto stat_iterator : term_iterator.second.term_change_stats) {
        stats_array.push_back({
          {"state", stat_iterator.first.c_str()},
          {"count", stat_iterator.second.count()},
          {"time_median_ms", stat_iterator.second.median()*1000},
          {"time_90_ms", stat_iterator.second.percentile(0.9)*1000},
        });
      }
      term_array.push_back({
        {"name", term_name.value()},
        {"uid", term_iterator.first},
        {"type", term_type_name.value()},
        {"data", stats_array}
      });
    }

    json_array.push_back({
      {"name", entity_name.value()},
      {"uid", entity_iterator.first},
      {"data", term_array}
    });
  }
  GXF_LOG_INFO("JobStatistics::getSchedulingTermStatistics Leaves");
  return json_array.dump();
}

// Request handler for "stat" service
Expected<std::string> JobStatistics::onGetStatistics(const std::string& resource) {
  // to locate the statistic item, type and id are specified, among which later
  // is optional.
  // when uid is missing statistics of all the instances of given type are collected
  gxf_uid_t uid = kUnspecifiedUid;  // by default, collect statistics for all the instances
  std::string item = resource;
  auto end = resource.find("/");
  if (end != std::string::npos) {
    item = resource.substr(0, end);
    auto uid_str = resource.substr(end+1);
    try {
      uid = std::stoll(uid_str);
    } catch (const std::invalid_argument&) {
      // conversion failed
      return Unexpected{GXF_ARGUMENT_INVALID};
    }
  }

  if (item == "entity") {
    return getEntityStatistics(uid);
  } else if (item == "codelet") {
    return getCodeletStatistics(uid);
  } else if (item == "event") {
    return getSchedulingEventStatistics(uid);
  } else if (item == "term") {
    return getSchedulingTermStatistics(uid);
  }
  return Unexpected{GXF_ARGUMENT_INVALID};
}

}  // namespace gxf
}  // namespace nvidia
