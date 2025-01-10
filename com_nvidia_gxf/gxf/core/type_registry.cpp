/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <mutex>
#include <string>

#include "yaml-cpp/yaml.h"

#include "gxf/core/type_registry.hpp"

namespace nvidia {
namespace gxf {

Expected<void> TypeRegistry::add(gxf_tid_t tid, const char* component_type_name) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  if (tids_.find(component_type_name) != tids_.end()) {
    GXF_LOG_ERROR("Trying to add duplicate component type %s.", component_type_name);
    return Unexpected{GXF_FACTORY_DUPLICATE_TID};
  }
  if (names_.find(tid) != names_.end()) {
    const char* component_type_name = names_[tid].c_str();
    GXF_LOG_ERROR("Trying to add duplicate component type (id=%016lx%016lx) %s.",
    tid.hash1, tid.hash2, component_type_name);
    return Unexpected{GXF_FACTORY_DUPLICATE_TID};
  }
  tids_[std::string(component_type_name)] = tid;
  names_[tid] = std::string(component_type_name);
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
    GXF_LOG_WARNING("Unknown type: %s", component_type_name);
    return Unexpected{GXF_FACTORY_UNKNOWN_CLASS_NAME};
  }

  return it->second;
}

Expected<bool> TypeRegistry::is_base(gxf_tid_t derived, gxf_tid_t base) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  if (names_.find(derived) == names_.end()) {
    GXF_LOG_ERROR("Component with TID 0x%016zx%016zx, not found", derived.hash1, derived.hash2);
    return Unexpected{GXF_QUERY_NOT_FOUND};
  }

  if (names_.find(base) == names_.end()) {
    GXF_LOG_ERROR("Component with TID 0x%016zx%016zx, not found", base.hash1, base.hash2);
    return Unexpected{GXF_QUERY_NOT_FOUND};
  }

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
    auto base_result = is_base(middle, base);
    if (!base_result) { return ForwardError(base_result); }
    if (base_result.value()) return true;
  }

  return false;
}

Expected<const char*> TypeRegistry::name(gxf_tid_t tid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  const auto it = names_.find(tid);
  if (it == names_.end()) {
    // Skip error log for null tid. Null tid is used for internal setup,
    // hence we dont want to log it as an error
    if (!GxfTidIsNull(tid)) {
      GXF_LOG_ERROR("Component with TID 0x%016zx%016zx, not found", tid.hash1, tid.hash2);
    }
    return Unexpected{GXF_QUERY_NOT_FOUND};
  }

  return it->second.c_str();
}

}  // namespace gxf
}  // namespace nvidia
