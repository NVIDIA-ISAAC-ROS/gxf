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

#include "gxf/app/entity_group.hpp"
#include <vector>
namespace nvidia {
namespace gxf {

Expected<void> EntityGroup::setup(gxf_context_t context, const char* name) {
  auto result = GxfCreateEntityGroup(context, name, &gid_);
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create entity group [%s] with error %s", name, GxfResultStr(result));
    return Unexpected{result};
  }
  name_ = name;
  return Success;
}

Expected<void> EntityGroup::add(GraphEntityPtr entity) {
  if (entity_members_.find(entity->name()) != entity_members_.end()) {
    GXF_LOG_ERROR("GraphEntity with same name [%s] already exists in EntityGroup [%s]",
      entity->name(), name_.c_str());
    return Unexpected{GXF_FAILURE};
  }
  // Add each Entity into just created EntityGroup
  auto result = GxfUpdateEntityGroup(entity->context(), gid_, entity->eid());
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to add entity [name: %s, eid: %ld] to EntityGroup %s",
                  entity->name(), entity->eid(), name_.c_str());
    return Unexpected{result};
  }
  entity_members_.emplace(entity->name(), entity);
  return Success;
}

Expected<void> EntityGroup::add(std::vector<GraphEntityPtr> entity_members) {
  for (const auto& entity : entity_members) {
    GXF_RETURN_IF_ERROR(this->add(entity));
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
