/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/component_serializer.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t ComponentSerializer::serialize_component_abi(gxf_uid_t cid, Endpoint* endpoint,
                                                          uint64_t* size) {
  if (endpoint == nullptr || size == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  gxf_tid_t tid;
  void* component;
  return ToResultCode(
      ExpectedOrCode(GxfComponentType(context(), cid, &tid))
      .and_then([&](){
        return ExpectedOrCode(GxfComponentPointer(context(), cid, tid, &component));
      })
      .and_then([&](){ return getSerializer(tid); })
      .map([&](Serializer serializer){ return serializer(component, endpoint); })
      .map([&](size_t component_size) {
        *size = component_size;
        return Success;
      }));
}

gxf_result_t ComponentSerializer::deserialize_component_abi(gxf_uid_t cid, Endpoint* endpoint) {
  if (endpoint == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  gxf_tid_t tid;
  void* component;
  return ToResultCode(
      ExpectedOrCode(GxfComponentType(context(), cid, &tid))
      .and_then([&](){
        return ExpectedOrCode(GxfComponentPointer(context(), cid, tid, &component));
      })
      .and_then([&](){ return getDeserializer(tid); })
      .map([&](Deserializer deserializer){ return deserializer(component, endpoint); }));
}

Expected<size_t> ComponentSerializer::serializeComponent(UntypedHandle component,
                                                         Endpoint* endpoint) {
  size_t size;
  return ExpectedOrCode(serialize_component_abi(component.cid(), endpoint, &size), size);
}

Expected<void> ComponentSerializer::deserializeComponent(UntypedHandle component,
                                                         Endpoint* endpoint) {
  return ExpectedOrCode(deserialize_component_abi(component.cid(), endpoint));
}

Expected<ComponentSerializer::Serializer> ComponentSerializer::getSerializer(gxf_tid_t tid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto entry = serializer_map_.find(tid);
  if (entry == serializer_map_.end() || entry->second.serializer == nullptr) {
    GXF_LOG_VERBOSE("Serializer not found for TID 0x%016zx%016zx", tid.hash1, tid.hash2);
    return Unexpected{GXF_FAILURE};
  }
  return entry->second.serializer;
}

Expected<ComponentSerializer::Deserializer>
ComponentSerializer::getDeserializer(gxf_tid_t tid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto entry = serializer_map_.find(tid);
  if (entry == serializer_map_.end() || entry->second.deserializer == nullptr) {
    GXF_LOG_VERBOSE("Deserializer not found for TID 0x%016zx%016zx", tid.hash1, tid.hash2);
    return Unexpected{GXF_FAILURE};
  }
  return entry->second.deserializer;
}

Expected<void> ComponentSerializer::setSerializer(gxf_tid_t tid, Serializer serializer) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto result = serializer_map_.insert({tid, SerializerFunctions{serializer, nullptr}});
  if (!result.second) {
    if (result.first->second.serializer != nullptr) {
      GXF_LOG_ERROR("Failed to set serializer for TID 0x%016zx%016zx", tid.hash1, tid.hash2);
      return Unexpected{GXF_FAILURE};
    }
    result.first->second.serializer = serializer;
  }
  return Success;
}

Expected<void> ComponentSerializer::setDeserializer(gxf_tid_t tid, Deserializer deserializer) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto result = serializer_map_.insert({tid, SerializerFunctions{nullptr, deserializer}});
  if (!result.second) {
    if (result.first->second.deserializer != nullptr) {
      GXF_LOG_ERROR("Failed to set deserializer for TID 0x%016zx%016zx", tid.hash1, tid.hash2);
      return Unexpected{GXF_FAILURE};
    }
    result.first->second.deserializer = deserializer;
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
