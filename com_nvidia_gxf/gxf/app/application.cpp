/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <string>
#include <vector>

// Disable breakpad for QNX
// execinfo.h is not available in the QNX toolchain
#ifdef __linux__
#include "client/linux/handler/exception_handler.h"
#include "common/backtrace.hpp"
#endif

#include "gxf/app/application.hpp"
#include "gxf/app/graph_utils.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/ucx/ucx_common.hpp"
#include "gxf/ucx/ucx_component_serializer.hpp"
#include "gxf/ucx/ucx_entity_serializer.hpp"
#include "gxf/ucx/ucx_receiver.hpp"
#include "gxf/ucx/ucx_serialization_buffer.hpp"
#include "gxf/ucx/ucx_transmitter.hpp"

namespace nvidia {
namespace gxf {

// execinfo.h is not available in the QNX toolchain
#ifdef __linux__
namespace {
  // Anonymous namespace within nvidia::gxf
  // no need to add static keyword within anonymous namespace which creates internal linkage
  bool onMinidump(const google_breakpad::MinidumpDescriptor& descriptor,
    void* context, bool succeeded) {
    // Print header
    std::fprintf(stderr, "\033[1;31m");
    std::fprintf(stderr, "====================================================================================================\n");  // NOLINT
    std::fprintf(stderr, "|                            GXF terminated unexpectedly                                           |\n");  // NOLINT
    std::fprintf(stderr, "====================================================================================================\n");  // NOLINT
    std::fprintf(stderr, "\033[0m");

    PrettyPrintBacktrace();

    // Print footer with mention to minidump
    std::fprintf(stderr, "\033[1;31m");
    std::fprintf(stderr, "====================================================================================================\n");  // NOLINT
    std::fprintf(stderr, "Minidump written to: %s\n", descriptor.path());
    std::fprintf(stderr, "\033[0m");
    return succeeded;
  }
}  // namespace
#endif  // __linux__

Expected<void> Application::setupCrashHandler() {
  #ifdef __linux__
  static google_breakpad::MinidumpDescriptor descriptor("/tmp");
  static google_breakpad::ExceptionHandler eh(descriptor, NULL, onMinidump, NULL, true, -1);
  #else
  GXF_LOG_WARNING("This is a GXF unsupported platform. "
    "In case of any failures, no crash logs will be reported");
  #endif
  return Success;
}

Application::Application() {
  setupCrashHandler();
  config_parser_ = std::make_unique<ConfigParser>();
  gxf_result_t result = GxfContextCreate(&context_);
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Failed to create Gxf Context with error %s", GxfResultStr(result));
    return;
  }

  GxfSetSeverity(context_, GXF_SEVERITY_INFO);

  runtime_ext_ = std::make_shared<DefaultExtension>();
  runtime_ext_->setInfo(generate_tid(), "RuntimeExtension",
                        "Extension used to register components at runtime", "NVIDIA", "0.0.1",
                        "NVIDIA");
  runtime_ext_->setDisplayInfo("Runtime Extension", "Runtime", "GXF Runtime Extension");
  result = GxfLoadExtensionFromPointer(context_, static_cast<void*>(runtime_ext_.get()));
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Failed to register application runtime extension with error %s",
                  GxfResultStr(result));
  }
  next_ucx_port_ = DEFAULT_UCX_PORT;

  // Enable UCX environment settings
  // If UCX_PROTO_ENABLE is not already set, set it to enable UCX Protocols v2
  setenv("UCX_PROTO_ENABLE", "y", 0);
  // Reuse address
  // (see https://github.com/openucx/ucx/issues/8585 and https://github.com/rapidsai/ucxx#c-1)
  setenv("UCX_TCP_CM_REUSEADDR", "y", 0);
}

Application::~Application() {
  // GxfContextDestroy(context_);
  extension_manager_.unloadAll();
}

Expected<void> Application::loadExtensionManifest(const char* manifest) {
  std::filesystem::path filePath(manifest);
  if (!std::filesystem::exists(filePath)) {
    GXF_LOG_ERROR("Manifest file not found %s", manifest);
    return Unexpected{GXF_FILE_NOT_FOUND};
  }

  const GxfLoadExtensionsInfo info{nullptr, 0, &manifest, 1, nullptr};
  auto result = GxfLoadExtensions(context_, &info);
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Failed to load extensions from manifest %s", manifest);
    return Unexpected{result};
  }

  return extension_manager_.loadManifest(manifest);
}

Expected<void> Application::run() {
  commitCompose();

  auto result = checkConfiguration();
  if (!result) {
    GXF_LOG_ERROR("Incorrect application configuration!");
    return Unexpected{GXF_FAILURE};
  }

  result = finalize();
  if (!result) {
    GXF_LOG_ERROR("Failed to finalize application");
    return Unexpected{GXF_FAILURE};
  }

  result = activate();
  if (!result) {
    GXF_LOG_ERROR("Failed to activate application with error [%s]", GxfResultStr(result.error()));
    return result;
  }

  GXF_LOG_INFO("Running Application ....");
  if (mode_ == ExecutionMode::kMultiSegment) {
    for (auto& [name, segment] : segments_) { result &= segment->runAsync(); }
    for (auto& [name, segment] : segments_) { result &= segment->wait(); }
  } else {
    result.substitute(GxfGraphRun(context_));
  }

  if (!result) {
    GXF_LOG_ERROR("Failed to run application with error [%s]", GxfResultStr(result.error()));
    return result;
  }

  result = deactivate();
  if (!result) {
    GXF_LOG_ERROR("Failed to deactivate application with error [%s]", GxfResultStr(result.error()));
  }

  return result;
}

Expected<void> Application::runAsync() {
  auto result = checkConfiguration();
  if (!result) {
    GXF_LOG_ERROR("Incorrect application configuration!");
    return Unexpected{GXF_FAILURE};
  }

  result = finalize();
  if (!result) {
    GXF_LOG_ERROR("Failed to finalize application");
    return Unexpected{GXF_FAILURE};
  }

  result = activate();
  if (!result) {
    GXF_LOG_ERROR("Failed to activate application!");
    return result;
  }

  GXF_LOG_INFO("Running Application ....");
  if (mode_ == ExecutionMode::kMultiSegment) {
    for (auto& [name, segment] : segments_) { result &= segment->runAsync(); }
  } else {
    result.substitute(GxfGraphRunAsync(context_));
  }

  return result;
}

Expected<void> Application::interrupt() {
  GXF_LOG_INFO("Interrupting Application ....");
  Expected<void> result;
  if (mode_ == ExecutionMode::kMultiSegment) {
    for (auto& [name, segment] : segments_) { result &= segment->interrupt(); }
  } else {
    result.substitute(GxfGraphInterrupt(context_));
  }

  return result;
}

Expected<void> Application::wait() {
  GXF_LOG_INFO("Waiting Application ....");
  Expected<void> result;

  if (mode_ == ExecutionMode::kMultiSegment) {
    for (auto& [name, segment] : segments_) { result &= segment->wait(); }
  } else {
    result.substitute(GxfGraphWait(context_));
  }

  return result;
}

Expected<void> Application::activate() {
  GXF_LOG_INFO("Activating Application ....");
  Expected<void> result;
  if (mode_ == ExecutionMode::kMultiSegment) {
    for (auto& [name, segment] : segments_) { result &= segment->activate(); }
  } else {
    result.substitute(GxfGraphActivate(context_));
  }

  return result;
}

Expected<void> Application::deactivate() {
  GXF_LOG_INFO("Deactivating Application ....");
  Expected<void> result;
  if (mode_ == ExecutionMode::kMultiSegment) {
    for (auto& [name, segment] : segments_) { result &= segment->deactivate(); }
  } else {
    result.substitute(GxfGraphDeactivate(context_));
  }
  return result;
}

Expected<void> Application::finalize() {
  GXF_LOG_DEBUG("Finalizing Application ....");
  Expected<void> result;
  if (mode_ == ExecutionMode::kMultiSegment || mode_ == ExecutionMode::kDistributed) {
    for (auto& [name, segment] : segments_) { result &= segment->createNetworkContext(); }
  }

  return result;
}

Expected<void> Application::checkConfiguration() {
  if (!segments_.empty() && !entities_.empty()) {
    // We don't allow both segments and entities, except the entities being driver or worker
    // check the non empty entities:
    if ((driver_ == nullptr && worker_ == nullptr)) {
      // the entities_ are non driver nor worker, don't allow
      GXF_LOG_ERROR("An application can either create segments or entities but not both together");
      return Unexpected{GXF_ARGUMENT_INVALID};
    } else if (driver_ != nullptr && worker_ == nullptr) {
      // the entities_ is driver only
      GXF_LOG_INFO("Application is running with graph driver");
      mode_ = ExecutionMode::kDistributed;
    } else if (driver_ != nullptr && worker_ != nullptr) {
      // the entities_ are driver and worker
      GXF_LOG_INFO("Application is running with both graph worker and graph driver");
      mode_ = ExecutionMode::kDistributed;
    } else if (driver_ == nullptr && worker_ != nullptr) {
      // the entities_ is worker only
      GXF_LOG_INFO("Application is running with graph worker");
      mode_ = ExecutionMode::kDistributed;
    }
  } else {
    // segments_ only or entities_ only
    if (segments_.empty() && entities_.empty()) {
      GXF_LOG_ERROR("An application does not have segments nor entities");
      return Unexpected{GXF_ARGUMENT_INVALID};
    }
    if (segments_.empty()) {
      GXF_LOG_INFO("Application is running in kSingleSegment mode");
      mode_ = ExecutionMode::kSingleSegment;
    } else {
      GXF_LOG_INFO("Application is running in kMultiSegment mode");
      mode_ = ExecutionMode::kMultiSegment;
    }
  }

  auto result = Success;
  for (const auto& [name, segment] : segments_) { result &= segment->checkConfiguration(); }

  return result;
}

gxf_result_t Application::setSegmentSeverity(const char* name, gxf_severity_t severity) {
  auto segment_itr = segments_.find(name);
  if (segment_itr == segments_.end()) {
    GXF_LOG_ERROR("Segment [%s] not found", name);
    return GXF_ARGUMENT_INVALID;
  }

  return segment_itr->second->setSeverity(severity);
}

Expected<void> Application::connect(SegmentPtr source, SegmentPtr target,
                                    std::vector<SegmentPortPair> port_maps) {
  if (source == nullptr || target == nullptr || port_maps.empty()) {
    GXF_LOG_ERROR("connect() API requires valid source and target; non empty port_maps");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  for (auto& pair : port_maps) {
    pair.tx.segment = source->name();
    pair.rx.segment = target->name();
  }
  segment_connections_plan_.emplace(source, target, port_maps);
  return Success;
}

// 1. source != nullptr && target == nullptr
//    local source; remote target. ucx_port set in source becomes temporary,
//    and will be updated by GraphDriver remotely according to target ucx_port
// 2. source == nullptr && target != nullptr
//    remote source; local target. ucx_port set in target stays,
//    and later will be queried: local-Worker -> Driver -> remote-Worker
// 3. source != nullptr && target != nullptr
//    local source; local target. ucx_port set in both source and target stay,
//    do not need Worker & Driver to resolve ucx rx address
Expected<void> Application::commitConnect(SegmentPtr source, SegmentPtr target,
                                    std::vector<SegmentPortPair> port_maps) {
  for (auto map : port_maps) {
    if (source != nullptr) {
      map.tx.segment = source->name();
    } else {
      GXF_LOG_DEBUG("nullptr source segment for port_map: [%s -> %s]",
        map.tx.to_string().c_str(), map.rx.to_string().c_str());
    }
    if (target != nullptr) {
      map.rx.segment = target->name();
    } else {
      GXF_LOG_DEBUG("nullptr target segment for port_map: [%s -> %s]",
        map.tx.to_string().c_str(), map.rx.to_string().c_str());
    }
    segment_connections_.emplace(source, target, port_maps);
  }
  for (const auto& map : port_maps) {
    GXF_LOG_DEBUG("Attempting to commitConnect Tx [%s] from [%s] with Rx [%s] from [%s]",
                  map.tx.queue.c_str(), map.tx.entity.c_str(), map.rx.queue.c_str(),
                  map.rx.entity.c_str());

    if (source != nullptr) {
      // Setup the transmitter entity
      auto tx_entity = source->getEntity(map.tx.entity.c_str());
      Handle<Allocator> allocator;
      auto maybe_allocator = tx_entity->try_get<UnboundedAllocator>(map.tx.queue.c_str());
      if (!maybe_allocator) {
        allocator = tx_entity->add<UnboundedAllocator>((map.tx.queue.c_str()));
      } else {
        allocator = maybe_allocator.value();
      }

      Handle<UcxSerializationBuffer> buffer;
      auto maybe_buffer = tx_entity->try_get<UcxSerializationBuffer>(map.tx.queue.c_str());
      if (!maybe_buffer) {
        buffer = tx_entity->add<UcxSerializationBuffer>(map.tx.queue.c_str());
        RETURN_IF_ERROR(buffer->set_allocator(allocator));
      } else {
        buffer = maybe_buffer.value();
      }

      // auto transmitter = tx_entity->addTransmitter<UcxTransmitter>(map.tx.queue.c_str());
      // Manually add the transmitter and update the codelet parameter since Ucx Tx
      // does not need a downstream receptive term to schedule the entity
      auto tx_name = UNWRAP_OR_RETURN(tx_entity->formatTxName(map.tx.queue.c_str()));
      auto transmitter = tx_entity->add<UcxTransmitter>(tx_name.c_str());

      std::string full_name = std::string(tx_entity->name()) + "/" + tx_name;
      auto result = tx_entity->updatePort(map.tx.queue.c_str(), full_name);
      RETURN_IF_ERROR(transmitter->set_serialization_buffer(buffer));
      RETURN_IF_ERROR(transmitter->set_port(next_ucx_port_));
    }

    if (target != nullptr) {
      auto rx_entity = target->getEntity(map.rx.entity.c_str());
      Handle<Allocator> allocator;
      auto maybe_allocator = rx_entity->try_get<UnboundedAllocator>(map.rx.queue.c_str());
      if (!maybe_allocator) {
        allocator = rx_entity->add<UnboundedAllocator>((map.rx.queue.c_str()));
      } else {
        allocator = maybe_allocator.value();
      }

      Handle<UcxSerializationBuffer> buffer;
      auto maybe_buffer = rx_entity->try_get<UcxSerializationBuffer>(map.rx.queue.c_str());
      if (!maybe_buffer) {
        buffer = rx_entity->add<UcxSerializationBuffer>(map.rx.queue.c_str());
        RETURN_IF_ERROR(buffer->set_allocator(allocator));
      } else {
        buffer = maybe_buffer.value();
      }

      auto receiver = rx_entity->addReceiver<UcxReceiver>(map.rx.queue.c_str());
      RETURN_IF_ERROR(receiver->set_serialization_buffer(buffer));
      RETURN_IF_ERROR(receiver->set_port(next_ucx_port_));
    }

    if (source != nullptr || target != nullptr) {
      next_ucx_port_++;
    }
  }

  return Success;
}

Expected<void> Application::commitSegment(SegmentPtr segment, const char* name) {
  gxf_context_t segment_context = kNullContext;
  gxf_result_t result = GxfContextCreate(&segment_context);
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Failed to create new context for segment %s", name);
    return Unexpected{result};
  }
  auto maybe = extension_manager_.registerExtensions(segment_context);
  if (!maybe) {  // dep -> compose()
    GXF_LOG_ERROR("Failed to register extensions to segment [%s]", name);
    return maybe;
  }

  segment->setup(segment_context, name, runtime_ext_);
  segment->setSeverity(GXF_SEVERITY_INFO);
  segment->compose();  // dep -> commitConnect()
  segments_.emplace(name, segment);

  return Success;
}

Expected<void> Application::commitCompose() {
  // createSegment() populates segments_plan_
  if (segments_plan_.empty()) {
    GXF_LOG_DEBUG("Application %s has no create segments plan", name_.c_str());
    return Success;
  }
  // connect() populates segment_connections_plan_
  if (segment_connections_plan_.empty()) {
    GXF_LOG_DEBUG("Application %s has no connect segments plan", name_.c_str());
  }

  // get segment control config
  auto maybe_segment_control = config_parser_->getSegmentConfig();
  if (!maybe_segment_control) {
    GXF_LOG_ERROR("Config parser failed to get segment config");
    return Unexpected{maybe_segment_control.error()};
  }
  auto segment_control = maybe_segment_control.value();
  if (maybe_segment_control.value() == nullptr) {
    GXF_LOG_ERROR("invalid pointer to segment control");
    return Unexpected{GXF_FAILURE};
  }

  if (segment_control->enable_all_segments) {
    for (const auto& it : segments_plan_) {
      enabled_segments_.emplace(it.first);
      auto maybe = commitSegment(it.second, it.first.c_str());
      if (!maybe) {
        GXF_LOG_ERROR("Failed to commit segment %s", it.first.c_str());
        return maybe;
      }
    }
  } else {
    // populate enabled segments from the control config
    for (const auto& name : segment_control->enabled_segments.names) {
      GXF_LOG_DEBUG("Enable segment name: %s", name.c_str());
      enabled_segments_.emplace(name);
    }

    // commit the plan for creating segments
    for (const auto& it : segments_plan_) {
      if (enabled_segments_.find(it.first) != enabled_segments_.end()) {
        GXF_LOG_DEBUG("Committing segment %s", it.first.c_str());
        auto maybe = commitSegment(it.second, it.first.c_str());
        if (!maybe) {
          GXF_LOG_ERROR("Failed to commit segment %s", it.first.c_str());
          return maybe;
        }
      }
    }
  }

  // commit the plan for connection between segments
  //    1. segment contains UCX Tx only;
  //    2. segment contains UCX Rx only;
  //    3. segment contains UCX Rx and UCX Tx.
  for (const auto& entry : segment_connections_plan_) {
    SegmentPtr source, target = nullptr;
    if (enabled_segments_.find(entry.source->name()) != enabled_segments_.end()) {
      source = entry.source;
    }
    if (enabled_segments_.find(entry.target->name()) != enabled_segments_.end()) {
      target = entry.target;
    }
    auto maybe = commitConnect(source, target, entry.port_maps);
    if (!maybe) {
      GXF_LOG_ERROR("Failed to commit connection [source: %s, target: %s]",
        entry.source->name(), entry.target->name());
      return maybe;
    }
  }

  // commit driver according to segment control config
  // condition: only if user manually add driver
  if (segment_control->driver.enabled) {  // need a driver
    // if config ask to add a driver and not set yet by C++ API
    if (driver_ == nullptr) {  // auto-add a driver
      std::string name = "default_driver";
      if (!segment_control->driver.name.empty()) {
        name = segment_control->driver.name;
      }
      // create driver wrapper according to user config
      driver_ = std::make_shared<Driver>(this, name);
    } else {
      GXF_LOG_DEBUG("Driver is already set in compose() as binary, by calling C++ API");
    }
    // update worker parameters if config provides
    if (segment_control->driver.port > 0) {
      GXF_LOG_DEBUG("Setting driver port to be %d", segment_control->driver.port);
      driver_->setPort(segment_control->driver.port);
    }
    // commit driver plan
    driver_->commit();
  }

  // commit worker according to segment control config
  // condition:
  //   1. segment control enable non zero segments; AND
  //   2. segment control enable a subset of segments in plan; AND
  //   3. segment control does not disable worker; AND
  if (!enabled_segments_.empty() &&
      enabled_segments_.size() < segments_plan_.size() &&
      segment_control->worker.enabled) {  // need a worker
    if (worker_ == nullptr) {  // auto-add a worker
      std::string name = "default_worker";
      if (!segment_control->worker.name.empty()) {
        name = segment_control->worker.name;
      }
      // help user to auto-create worker wrapper in app
      worker_ = std::make_shared<Worker>(this, name);
    } else {
      GXF_LOG_DEBUG("Worker is already set in compose() as binary, by calling C++ API");
    }
    // update worker parameters if config provides
    if (segment_control->worker.port > 0) {
      GXF_LOG_DEBUG("Setting worker port to be %d", segment_control->worker.port);
      worker_->setPort(segment_control->worker.port);
    }
    if (!segment_control->worker.driver_ip.empty()) {
      GXF_LOG_DEBUG("Setting worker driver_ip to be %s",
        segment_control->worker.driver_ip.c_str());
      worker_->setDriverIp(segment_control->worker.driver_ip);
    }
    if (segment_control->worker.driver_port > 0) {
      GXF_LOG_DEBUG("Setting worker driver_port to be %d", segment_control->worker.driver_port);
      worker_->setDriverPort(segment_control->worker.driver_port);
    }
    if (!enabled_segments_.empty()) {
      // if C++ API setWorker() was used to set segments, config will overwrite
      // Priority(config) > Priority(setWorker())
      worker_->setSegments(enabled_segments_);
    }
    // commit worker plan
    worker_->commit();
  } else {
    if (enabled_segments_.empty() && segment_control->worker.enabled) {
      GXF_LOG_INFO("No segments enabled, skip adding a worker");
    }
    if (enabled_segments_.size() == segments_plan_.size() && segment_control->worker.enabled) {
      GXF_LOG_INFO("All segments enabled in one process, skip adding a worker");
    }
  }

  return Success;
}

Expected<void> Application::setConfig(const std::string& file_path) {
  config_parser_->setFilePath(file_path);
  return Success;
}

Expected<void> Application::setConfig(int argc, char** argv) {
  config_parser_->setFilePath(argc, argv);
  return Success;
}


}  // namespace gxf
}  // namespace nvidia
