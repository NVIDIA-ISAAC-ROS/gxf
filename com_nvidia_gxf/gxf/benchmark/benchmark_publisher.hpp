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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gxf/benchmark/entity_buffer.hpp"

#include "gxf/std/codelet.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// A benchmark publisher that publishes buffered benchmark messages
class BenchmarkPublisher : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

  // Getter of the underlying entity buffer component
  gxf::Handle<EntityBuffer> getEntityBuffer();

  // Getter of the recorded published timestamps
  std::vector<std::chrono::nanoseconds>& getPublishedTimestamps();

  // Getter of the associated async scheduling term
  gxf::Handle<gxf::AsynchronousSchedulingTerm> getAsyncSchedulingterm();

  // Setter of the number of benchmark messages to publish
  // 0 means no limit
  void setNumOfMessagesToPublish(uint64_t num_of_messages_to_publish);

  // Clear the runtime state
  // Calling this fucntion is sufficient to reset state for a new benchmark iteration
  void clearRecordedTimestamps();

 private:
  gxf::Parameter<gxf::Handle<EntityBuffer>> entity_buffer_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> transmitter_;
  gxf::Parameter<gxf::Handle<gxf::AsynchronousSchedulingTerm>>
      benchmark_publisher_async_scheduling_term_;

  // Messages published timestamps
  std::vector<std::chrono::nanoseconds> published_timestamps_;

  // The number of messages to publish for benchmarking
  uint64_t num_of_messages_to_publish_;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
