/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_GXF_APPLICATION_WORKER_HPP_
#define NVIDIA_GXF_APPLICATION_WORKER_HPP_

#include <memory>
#include <string>
#include <unordered_set>

#include "gxf/app/segment.hpp"
#include "gxf/std/graph_worker.hpp"

namespace nvidia {
namespace gxf {

class Application;

/**
 * @brief GraphWorker representation in Application API layer
 * It manages the execution of selected segments
 *
 */
class Worker {
 public:
  Worker(Application* owner, const std::string& name);

  std::string name() { return name_; }
  Expected<void> setPort(uint32_t port) { port_ = port; return Success; }
  Expected<void> setDriverIp(const std::string& ip) { driver_ip_ = ip; return Success; }
  Expected<void> setDriverPort(uint32_t port) { driver_port_ = port; return Success; }
  // set selected segments to run
  Expected<void> setSegments(const std::unordered_set<std::string>& segment_names) {
    segment_names_ = segment_names;
    return Success;
  }
  // commit the setup into GraphWorker
  Expected<void> commit();

 private:
  Application* owner_;
  std::string name_;
  GraphEntityPtr worker_entity_;
  // core component
  Handle<GraphWorker> graph_worker_;
  // server interface object
  Handle<IPCServer> server_;
  // client interface object
  Handle<IPCClient> client_;

  // worker server's own port
  uint32_t port_ = 0;
  // target driver server's IP address
  std::string driver_ip_;
  // target driver server's port
  uint32_t driver_port_ = 0;
  // selected segments to run by this worker
  std::unordered_set<std::string> segment_names_;
};
typedef std::shared_ptr<Worker> WorkerPtr;

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_APPLICATION_WORKER_HPP_
