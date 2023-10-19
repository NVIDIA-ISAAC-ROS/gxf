/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/std_entity_id_serializer.hpp"

#if defined(_QNX_SOURCE)
#include <net/netbyte.h>
#else
#include <endian.h>
#endif
#include <cstring>
#include <string>

namespace nvidia {
namespace gxf {

namespace {

// Serializes EntityHeader
Expected<size_t> SerializeEntityHeader(
    StdEntityIdSerializer::EntityHeader header, Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  header.entity_id = htole64(header.entity_id);
  header.sequence_number = htole64(header.sequence_number);
  return endpoint->writeTrivialType(&header).substitute(sizeof(header));
}

// Deserializes EntityHeader
Expected<StdEntityIdSerializer::EntityHeader> DeserializeEntityHeader(
    Endpoint* endpoint) {
  if (!endpoint) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  StdEntityIdSerializer::EntityHeader header;
  return endpoint->readTrivialType(&header).and_then([&]() {
    header.entity_id = le64toh(header.entity_id);
    return header;
  });
}

}  // namespace

gxf_result_t StdEntityIdSerializer::serialize_entity_abi(gxf_uid_t eid,
                                                         Endpoint* endpoint,
                                                         uint64_t* size) {
  if (endpoint == nullptr || size == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  GxfEntityRefCountInc(context(), eid);
  return gxf::ToResultCode(
      Entity::Shared(context(), eid)
          .map([&](Entity entity) { return entity.findAll(); })
          .and_then([&]() {
            EntityHeader entity_header;
            entity_header.entity_id = eid;
            entity_header.sequence_number = outgoing_sequence_number_++;
            return SerializeEntityHeader(entity_header, endpoint);
          })
          .assign_to(*size));
}

gxf_result_t StdEntityIdSerializer::deserialize_entity_abi(gxf_uid_t eid,
                                                           Endpoint* endpoint) {
  return GXF_FAILURE;
}

Expected<Entity> StdEntityIdSerializer::deserialize_entity_header_abi(
    Endpoint* endpoint) {
  if (endpoint == nullptr) {
    return Unexpected(GXF_ARGUMENT_NULL);
  }
  Expected<StdEntityIdSerializer::EntityHeader> entity_header =
      DeserializeEntityHeader(endpoint);
  if (incoming_sequence_number_++ != entity_header.value().sequence_number)
    GXF_LOG_ERROR("Sequence number does not match");

  const Expected<Entity> maybe_entity =
      Entity::Shared(context(), entity_header.value().entity_id);
  GxfEntityRefCountDec(context(), entity_header.value().entity_id);
  return maybe_entity;
}
}  // namespace gxf
}  // namespace nvidia
