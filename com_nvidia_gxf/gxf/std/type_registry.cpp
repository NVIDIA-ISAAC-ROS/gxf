/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string>

#include "yaml-cpp/yaml.h"

#include "gxf/std/type_registry.hpp"

namespace nvidia {
namespace gxf {

Expected<void> TypeRegistry::add(gxf_tid_t tid, const char* component_type_name) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  if (tids_.find(component_type_name) != tids_.end()) {
    return Unexpected{GXF_FACTORY_DUPLICATE_TID};
  }
  tids_[std::string(component_type_name)] = tid;
  return Success;
}

Expected<void> TypeRegistry::add_base(const char* component_type_name, const char* base_type_name) {
  const Expected<gxf_tid_t> component_tid = id_from_name(component_type_name);
  if (!component_tid) return ForwardError(component_tid);

  const Expected<gxf_tid_t> base_tid = id_from_name(base_type_name);
  if (!base_tid) return ForwardError(base_tid);

  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  bases_[component_tid.value()].insert(base_tid.value());

  return Success;
}

Expected<gxf_tid_t> TypeRegistry::id_from_name(const char* component_type_name) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  const auto it = tids_.find(std::string(component_type_name));
  if (it == tids_.end()) {
    GXF_LOG_ERROR("Unknown type: %s", component_type_name);
    return Unexpected{GXF_FACTORY_UNKNOWN_CLASS_NAME};
  }

  return it->second;
}

bool TypeRegistry::is_base(gxf_tid_t derived, gxf_tid_t base) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  // check if the type has any bases
  const auto it = bases_.find(derived);
  if (it == bases_.end()) {
    return false;
  }

  // check if it's a direct base
  if (it->second.count(base) > 0) {
    return true;
  }

  // recurse
  for (const auto& middle : it->second) {
    if (is_base(middle, base)) return true;
  }

  return false;
}

Expected<const char*> TypeRegistry::name(gxf_tid_t tid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  for (const auto& kvp : tids_) {
    if (kvp.second == tid) return kvp.first.c_str();
  }
  return Unexpected{GXF_FAILURE};
}

}  // namespace gxf
}  // namespace nvidia
