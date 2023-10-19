/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/entity_serializer.hpp"

namespace nvidia {
namespace gxf {

Expected<size_t> EntitySerializer::serializeEntity(Entity entity, Endpoint* endpoint) {
  size_t size;
  return ExpectedOrCode(serialize_entity_abi(entity.eid(), endpoint, &size), size);
}

Expected<void> EntitySerializer::deserializeEntity(Entity entity, Endpoint* endpoint) {
  return ExpectedOrCode(deserialize_entity_abi(entity.eid(), endpoint));
}

Expected<Entity> EntitySerializer::deserializeEntity(gxf_context_t context,
                                                     Endpoint* endpoint) {
  const Expected<Entity> maybe_entity = deserialize_entity_header_abi(endpoint);
  if (!maybe_entity) {
    return ForwardError(maybe_entity);
  }
  return maybe_entity;
}

}  // namespace gxf
}  // namespace nvidia
