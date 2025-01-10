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
#include "gxf/benchmark/benchmark_publisher.hpp"

#include <vector>

namespace nvidia {
namespace gxf {
namespace benchmark {

gxf_result_t BenchmarkPublisher::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      transmitter_, "transmitter",
      "Benchmark publisher transmitter",
      "Transmitter to publish benchmark messages");
  result &= registrar->parameter(
      entity_buffer_, "entity_buffer",
      "Benchmark message entity buffer",
      "Component that holds buffered benchmark message entities");
  result &= registrar->parameter(
      benchmark_publisher_async_scheduling_term_,
      "benchmark_publisher_async_scheduling_term",
      "Benchmark publisher execution control scheduling term",
      "A async scheduling term to control execution of the benchmark publisher");
  return gxf::ToResultCode(result);
}

gxf_result_t BenchmarkPublisher::start() {
  clearRecordedTimestamps();
  num_of_messages_to_publish_ = 0;
  return GXF_SUCCESS;
}

gxf_result_t BenchmarkPublisher::tick() {
  if (entity_buffer_->getBuffer().size() == 0) {
    GXF_LOG_WARNING("No message was present in the underlying entity buffer");
    return GXF_SUCCESS;
  }
  const size_t published_message_count = published_timestamps_.size();

  if (num_of_messages_to_publish_ != 0 &&
      published_message_count >= num_of_messages_to_publish_) {
    GXF_LOG_ERROR("Done publishing all %ld messages", published_message_count);
    benchmark_publisher_async_scheduling_term_->setEventState(
        nvidia::gxf::AsynchronousEventState::WAIT);
  } else {
    const size_t actual_buffer_data_index =
        published_message_count % entity_buffer_->getBuffer().size();
    transmitter_->publish(entity_buffer_->getBuffer().at(actual_buffer_data_index));
    published_timestamps_.push_back(
        std::chrono::nanoseconds(getExecutionTimestamp()));
  }
  return GXF_SUCCESS;
}

gxf::Handle<EntityBuffer> BenchmarkPublisher::getEntityBuffer() {
  return entity_buffer_.get();
}

std::vector<std::chrono::nanoseconds>& BenchmarkPublisher::getPublishedTimestamps() {
  return published_timestamps_;
}

gxf::Handle<gxf::AsynchronousSchedulingTerm> BenchmarkPublisher::getAsyncSchedulingterm() {
  return benchmark_publisher_async_scheduling_term_.get();
}

void BenchmarkPublisher::setNumOfMessagesToPublish(uint64_t num_of_messages_to_publish) {
  GXF_LOG_DEBUG("Set the number of messages to publish to to %ld",
                num_of_messages_to_publish);
  num_of_messages_to_publish_ = num_of_messages_to_publish;
}

void BenchmarkPublisher::clearRecordedTimestamps() {
  const size_t timestamp_count = published_timestamps_.size();
  published_timestamps_.clear();
  if (timestamp_count > 0) {
    published_timestamps_.reserve(timestamp_count*1.5);
  }
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
