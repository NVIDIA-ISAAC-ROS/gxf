/*
 * SPDX-FileCopyrightText: Copyright (c) 202-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_GXF_STD_QUEUE_HPP
#define NVIDIA_GXF_STD_QUEUE_HPP

#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {

// Interface for storing entities in a queue
class Queue : public Component {
 public:
  virtual ~Queue() = default;

  // Gets the next (oldest) entity in the queue.
  virtual gxf_result_t pop_abi(gxf_uid_t* uid) = 0;

  // Adds a new entity to the queue. The receiver takes shared
  // ownership of the entity.
  virtual gxf_result_t push_abi(gxf_uid_t other) = 0;

  // Peeks the entity at given index on the main stage.
  virtual gxf_result_t peek_abi(gxf_uid_t* uid, int32_t index) = 0;

  // The total number of entities the queue can hold.
  virtual size_t capacity_abi() = 0;

  // The total number of entities the queue currently holds.
  virtual size_t size_abi() = 0;

  Expected<Entity> pop();

  Expected<void> push(const Entity& other);

  Expected<Entity> peek(int32_t index = 0);

  size_t capacity();

  size_t size();
};

}  // namespace gxf
}  // namespace nvidia

#endif
