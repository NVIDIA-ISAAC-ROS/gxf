/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/parameter_storage.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nvidia {
namespace gxf {

ParameterStorage::ParameterStorage(gxf_context_t context) : context_(context) {}

Expected<void> ParameterStorage::parse(gxf_uid_t uid, const char* key, const YAML::Node& node,
                                       const std::string& prefix) {
  ParameterBackendBase* pointer = nullptr;
  {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);

    const auto it = parameters_.find(uid);
    if (it == parameters_.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }

    const auto jt = it->second.find(key);
    if (jt == it->second.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }
    pointer = jt->second.get();
  }

  return pointer->parse(node, prefix);
}

Expected<YAML::Node> ParameterStorage::wrap(gxf_uid_t uid, const char* key) {
  ParameterBackendBase* pointer = nullptr;
  {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);

    const auto it = parameters_.find(uid);
    if (it == parameters_.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }

    const auto jt = it->second.find(key);
    if (jt == it->second.end()) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }
    pointer = jt->second.get();
  }

  return pointer->wrap();
}

Expected<void> ParameterStorage::setStr(gxf_uid_t uid, const char* key, const char* value) {
  return set(uid, key, std::string(value));
}

Expected<const char*> ParameterStorage::getStr(gxf_uid_t uid, const char* key) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return getValuePointer<std::string>(uid, key)
        .map([](const std::string* pointer) { return pointer->c_str(); });
}

Expected<void> ParameterStorage::setPath(gxf_uid_t uid, const char* key, const char* value) {
  return set(uid, key, FilePath(value));
}

Expected<const char*> ParameterStorage::getPath(gxf_uid_t uid, const char* key) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return getValuePointer<FilePath>(uid, key)
        .map([](const std::string* pointer) { return pointer->c_str(); });
}

Expected<void> ParameterStorage::setHandle(gxf_uid_t uid, const char* key, gxf_uid_t value) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  auto maybe_ptr = getBackendPointerImpl<HandleParameterBackend>(uid, key);
  if (!maybe_ptr) { return ForwardError(maybe_ptr); }
  auto* ptr = maybe_ptr.value();

  const auto code = ptr->set(value);
  if (!code) { return code; }

  ptr->writeToFrontend();  // FIXME(v1) Special treatment for codelet parameters
  return Success;
}

Expected<gxf_uid_t> ParameterStorage::getHandle(gxf_uid_t uid, const char* key) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return getBackendPointerImpl<HandleParameterBackend>(uid, key)
        .map([](const HandleParameterBackend* pointer) { return pointer->get(); });
}

Expected<void> ParameterStorage::setStrVector(gxf_uid_t uid, const char* key, const char** value,
                                              uint64_t length) {
  if (!value) {
    GXF_LOG_ERROR("Value for the parameter, %s, is null", key);
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  std::vector<std::string> value_;
  for (uint64_t i = 0; i < length; i++) {
    if (!value[i]) {
      GXF_LOG_ERROR("Value at index %ld is null for vector param: %s", i, key);
      return Unexpected{GXF_ARGUMENT_NULL};
    }
    value_.push_back(std::string(value[i]));
  }
  return set(uid, key, value_);
}

// Adds the given value to a parameter and returns the result. The parameter is initialized to
// 0 in case it does not exist.
Expected<int64_t> ParameterStorage::addGetInt64(gxf_uid_t uid, const char* key, int64_t delta) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  auto it = parameters_.find(uid);
  if (it == parameters_.end()) {
    std::map<std::string, std::unique_ptr<ParameterBackendBase>> tmp;
    it = parameters_.insert({uid, std::move(tmp)}).first;
  }

  auto jt = it->second.find(key);
  if (jt == it->second.end()) {
    auto ptr = std::make_unique<ParameterBackend<int64_t>>();
    ptr->context_ = context_;
    ptr->uid_ = uid;
    ptr->flags_ = GXF_PARAMETER_FLAGS_OPTIONAL | GXF_PARAMETER_FLAGS_DYNAMIC;
    ptr->is_dynamic_ = true;
    ptr->key_ = key;
    ptr->headline_ = key;
    ptr->description_ = "N/A";
    // std::unique_ptr<ParameterBackendBase> base = ptr;
    jt = it->second.insert({key, std::move(ptr)}).first;
  }

  auto* ptr = dynamic_cast<ParameterBackend<int64_t>*>(jt->second.get());
  if (ptr == nullptr) { return Unexpected{GXF_PARAMETER_INVALID_TYPE}; }

  const Expected<void> code = ptr->set(ptr->try_get().value_or(0) + delta);
  if (!code) { return ForwardError(code); }

  ptr->writeToFrontend();  // FIXME(v1) Special treatment for codelet parameters

  const auto maybe = ptr->try_get();
  if (!maybe) { return Unexpected{GXF_PARAMETER_NOT_FOUND}; }
  return *maybe;
}

// Returns true if all mandatory parameters are available
Expected<void> ParameterStorage::isAvailable() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  for (const auto& kvp : parameters_) {
    for (const auto& kvp2 : kvp.second) {
      if (!kvp2.second) { return Unexpected{GXF_FAILURE}; }
      if (kvp2.second->isMandatory() && !kvp2.second->isAvailable()) {
        const char* component_name = "UNKNOWN";
        GxfParameterGetStr(context_, kvp.first, kInternalNameParameterKey, &component_name);
        const char* entity_name = "UNKNOWN";
        gxf_uid_t eid;
        GxfComponentEntity(context_, kvp.first, &eid);
        GxfParameterGetStr(context_, eid, kInternalNameParameterKey, &entity_name);
        GXF_LOG_ERROR("Mandatory parameter \"%s\" not set in component \"%s\" entity \"%s\"",
                       kvp2.first.c_str(), component_name, entity_name);
        return Unexpected{GXF_PARAMETER_MANDATORY_NOT_SET};
      }
    }
  }
  return Success;
}

// Returns true if all mandatory parameters of a component are available
Expected<void> ParameterStorage::isAvailable(gxf_uid_t uid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  auto component = parameters_.find(uid);
  if (component == parameters_.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }

  for (const auto& kvp : component->second) {
    if (!kvp.second) { return Unexpected{GXF_FAILURE}; }
    if (kvp.second->isMandatory() && !kvp.second->isAvailable()) {
      const char* component_name = "UNKNOWN";
      GxfParameterGetStr(context_, uid, kInternalNameParameterKey, &component_name);
      const char* entity_name = "UNKNOWN";
      gxf_uid_t eid;
      GxfComponentEntity(context_, uid, &eid);
      GxfParameterGetStr(context_, eid, kInternalNameParameterKey, &entity_name);
      GXF_LOG_ERROR("Mandatory parameter \"%s\" not set in component \"%s\" entity \"%s\"",
                     kvp.second->key(), component_name, entity_name);
      return Unexpected{GXF_PARAMETER_MANDATORY_NOT_SET};
    }
  }

  return Success;
}

Expected<void> ParameterStorage::clearEntityParameters(gxf_uid_t eid) {
  {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);

    const auto it = parameters_.find(eid);
    if (it == parameters_.end()) {
      return Unexpected{GXF_PARAMETER_NOT_FOUND};
    }
    parameters_.erase(it);
    return Success;
  }
}

}  // namespace gxf
}  // namespace nvidia
