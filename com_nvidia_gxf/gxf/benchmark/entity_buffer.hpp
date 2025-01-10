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

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {

// A benchmark entity buffer that stores any incoming message entities
class EntityBuffer : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  // Getter for the actual entity buffer
  std::vector<gxf::Entity>& getBuffer();

  // Set the maximum number of entity messages to be buffered
  // 0 means that buffering has no limit, otherwise incoming messages are dropped
  // once the buffer is full.
  void setMaxBufferSize(size_t max_buffer_size);

  // Check if the buffer is fully loaded
  bool isBufferFull();

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;

  std::vector<gxf::Entity> entity_buffer_;
  size_t max_buffer_size_ = 0;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
