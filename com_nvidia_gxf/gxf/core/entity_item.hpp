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
#ifndef NVIDIA_GXF_CORE_ENTITY_ITEM_HPP_
#define NVIDIA_GXF_CORE_ENTITY_ITEM_HPP_

#include <atomic>
#include <shared_mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include "common/fixed_vector.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

// Forward declaration
class ComponentFactory;

/**
 * @brief Stores information about a single component within an entity
 *
 */
struct ComponentItem {
  gxf_uid_t cid;
  gxf_tid_t tid;
  void* raw_pointer;
  void* component_pointer;
  friend class Entity;
};

/**
 * @brief Various lifecycle stages of an entity item. It is used internally by
 * entity warden to keep track of state changes during graph execution
 *
 */
enum class Stage : std::int8_t {
  kUninitialized = 0,               // Entity has been created but components are not initialized
  kInitializationInProgress = 1,    // All the components are being initalized sequentially
  kInitialized = 2,                 // All the components in the entity have been initialized
  kDeinitializationInProgress = 3,  // All the components are being deinitialized sequentially
  kDestructionInProgress = 4,       // All components are being destroyed in the factory
  kDestroyed = 5,                   // Entity is completely reset
};

/**
 * @brief Stores information about a single entity item in the warden
 *
 */
struct EntityItem {
  std::atomic<Stage> stage;
  gxf_uid_t uid;  // id of this struct itself
  FixedVector<ComponentItem, kMaxComponents> components;
  std::unordered_map<std::string, gxf_uid_t> interface;
  gxf_uid_t gid = kUnspecifiedUid;  // entity group id

  gxf_result_t initialize();

  gxf_result_t deinitialize();

  gxf_result_t destroy(ComponentFactory* factory);

  gxf_result_t destroyComponent(ComponentFactory* factory, gxf_uid_t cid);

  mutable std::shared_mutex entity_item_mutex_;
};

}  // namespace gxf
}  // namespace nvidia
#endif  // NVIDIA_GXF_CORE_ENTITY_ITEM_HPP_
