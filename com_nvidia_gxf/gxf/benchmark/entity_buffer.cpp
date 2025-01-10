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
#include "gxf/benchmark/entity_buffer.hpp"

#include <vector>

namespace nvidia {
namespace gxf {
namespace benchmark {

gxf_result_t EntityBuffer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(receiver_, "receiver",
      "Entity buffer receiver",
      "Receiver to get message entities for buffering");
  return gxf::ToResultCode(result);
}

gxf_result_t EntityBuffer::tick() {
  auto message = receiver_->receive();
  if (!message) {
    return message.error();
  }

  if (isBufferFull()) {
    GXF_LOG_DEBUG("Dropped an incoming message due to a full buffer");
    return GXF_SUCCESS;
  }

  entity_buffer_.push_back(message.value());
  GXF_LOG_DEBUG("Buffered a message (buffered message count = %ld)", entity_buffer_.size());

  return GXF_SUCCESS;
}

std::vector<gxf::Entity>& EntityBuffer::getBuffer() {
  return entity_buffer_;
}

void EntityBuffer::setMaxBufferSize(size_t max_buffer_size) {
  max_buffer_size_ = max_buffer_size;
  GXF_LOG_DEBUG("Set max buffer size to %ld", max_buffer_size);
}

bool EntityBuffer::isBufferFull() {
  if (max_buffer_size_ <= 0) {
    return false;
  }
  return (entity_buffer_.size() >= max_buffer_size_);
}

gxf_result_t EntityBuffer::stop() {
  GXF_LOG_DEBUG("Stopping the entity buffer");
  entity_buffer_.clear();
  return GXF_SUCCESS;
}

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
