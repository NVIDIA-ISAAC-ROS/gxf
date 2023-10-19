/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/std_entity_serializer.hpp"

#if defined (_QNX_SOURCE)
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
Expected<size_t> SerializeEntityHeader(StdEntitySerializer::EntityHeader header,
                                       Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  header.serialized_size = htole64(header.serialized_size);
  header.checksum = htole32(header.checksum);
  header.sequence_number = htole64(header.sequence_number);
  header.flags = htole32(header.flags);
  header.component_count = htole64(header.component_count);
  header.reserved = htole64(header.reserved);
  return endpoint->writeTrivialType(&header).substitute(sizeof(header));
}

// Deserializes EntityHeader
Expected<StdEntitySerializer::EntityHeader> DeserializeEntityHeader(Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  StdEntitySerializer::EntityHeader header;
  return endpoint->readTrivialType(&header).and_then([&]() {
        header.serialized_size = le64toh(header.serialized_size);
        header.checksum = le32toh(header.checksum);
        header.sequence_number = le64toh(header.sequence_number);
        header.flags = le32toh(header.flags);
        header.component_count = le64toh(header.component_count);
        header.reserved = le64toh(header.reserved);
        return header;
      });
}

// Serializes ComponentHeader
Expected<size_t> SerializeComponentHeader(StdEntitySerializer::ComponentHeader header,
                                          Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  header.serialized_size = htole64(header.serialized_size);
  header.tid.hash1 = htole64(header.tid.hash1);
  header.tid.hash2 = htole64(header.tid.hash2);
  header.name_size = htole64(header.name_size);
  return endpoint->writeTrivialType(&header).substitute(sizeof(header));
}

// Deserializes ComponentHeader
Expected<StdEntitySerializer::ComponentHeader> DeserializeComponentHeader(Endpoint* endpoint) {
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }
  StdEntitySerializer::ComponentHeader header;
  return endpoint->readTrivialType(&header).and_then([&]() {
        header.serialized_size = le64toh(header.serialized_size);
        header.tid.hash1 = le64toh(header.tid.hash1);
        header.tid.hash2 = le64toh(header.tid.hash2);
        header.name_size = le64toh(header.name_size);
        return header;
      });
}

}  // namespace

struct StdEntitySerializer::ComponentEntry {
  ComponentHeader header = { 0, GxfTidNull(), 0 };
  UntypedHandle component = UntypedHandle::Null();
  Handle<ComponentSerializer> serializer = Handle<ComponentSerializer>::Null();
};

gxf_result_t StdEntitySerializer::registerInterface(Registrar* registrar) {
  if (registrar == nullptr) { return GXF_ARGUMENT_NULL; }
  Expected<void> result;
  result &= registrar->parameter(
      component_serializers_, "component_serializers", "Component serializers",
      "List of serializers for serializing and deserializing components");
  result &= registrar->parameter(
      verbose_warning_, "verbose_warning", "Verbose Warning",
      "Whether or to print verbose warning", true);
  return ToResultCode(result);
}

gxf_result_t StdEntitySerializer::serialize_entity_abi(gxf_uid_t eid, Endpoint* endpoint,
                                                       uint64_t* size) {
  if (endpoint == nullptr || size == nullptr) { return GXF_ARGUMENT_NULL; }
  FixedVector<ComponentEntry, kMaxComponents> entries;
  return gxf::ToResultCode(Entity::Shared(context(), eid)
      .map([&](Entity entity) { return entity.findAll(); })
      .map([&](FixedVector<UntypedHandle, kMaxComponents> components) {
        return createComponentEntries(components);
      })
      .assign_to(entries)
      .and_then([&]() {
          EntityHeader entity_header;
          entity_header.serialized_size = 0;  // How can we compute this before serializing?
          entity_header.checksum = 0x00000000;
          entity_header.sequence_number = outgoing_sequence_number_++;
          entity_header.flags = 0x00000000;
          entity_header.component_count = entries.size();
          entity_header.reserved = 0;
          return SerializeEntityHeader(entity_header, endpoint);
      })
      .assign_to(*size)
      .and_then([&]() { return serializeComponents(entries, endpoint); })
      .map([&](size_t serialized_size) { *size += serialized_size; }));
}

Expected<Entity> StdEntitySerializer::deserialize_entity_header_abi(
    Endpoint* endpoint) {
  Entity entity;
  auto result = Entity::New(context())
      .assign_to(entity)
      .and_then([&]() { return DeserializeEntityHeader(endpoint); })
      .map([&](EntityHeader entity_header) {
        if (entity_header.sequence_number != incoming_sequence_number_) {
          if (verbose_warning_.get()) {
            GXF_LOG_WARNING("Got message %zu but expected message %zu",
                            entity_header.sequence_number,
                            incoming_sequence_number_);
          }
          incoming_sequence_number_ = entity_header.sequence_number;
        }
        incoming_sequence_number_++;
        return deserializeComponents(entity_header.component_count, entity,
                                     endpoint);
      })
      .substitute(entity);
  if (!result) {
    GXF_LOG_ERROR("Deserialize entity header failed");
  }
  return result;
}

gxf_result_t StdEntitySerializer::deserialize_entity_abi(gxf_uid_t eid, Endpoint* endpoint) {
  if (endpoint == nullptr) { return GXF_ARGUMENT_NULL; }
  Entity entity;
  return gxf::ToResultCode(Entity::Shared(context(), eid)
      .assign_to(entity)
      .and_then([&]() { return DeserializeEntityHeader(endpoint); })
      .map([&](EntityHeader entity_header) {
        if (entity_header.sequence_number != incoming_sequence_number_) {
          if (verbose_warning_.get()) {
            GXF_LOG_WARNING("Got message %zu but expected message %zu",
                            entity_header.sequence_number, incoming_sequence_number_);
          }
          incoming_sequence_number_ = entity_header.sequence_number;
        }
        incoming_sequence_number_++;
        return deserializeComponents(entity_header.component_count, entity, endpoint);
      }));
}

Expected<FixedVector<StdEntitySerializer::ComponentEntry, kMaxComponents>>
StdEntitySerializer::createComponentEntries(
    const FixedVector<UntypedHandle, kMaxComponents>& components) {
  FixedVector<ComponentEntry, kMaxComponents> entries;
  for (size_t i = 0; i < components.size(); i++) {
    const auto component = components[i];
    if (!component) { return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE}; }

    // Check if component is serializable
    auto component_serializer = findComponentSerializer(component->tid());
    if (!component_serializer) {
      GXF_LOG_WARNING("No serializer found for component '%s' with type ID 0x%016zx%016zx",
                      component->name(), component->tid().hash1, component->tid().hash2);
      continue;
    }

    // Create component header
    ComponentHeader component_header;
    component_header.serialized_size = 0;  // How can we compute this before serializing?
    component_header.tid = component->tid();
    component_header.name_size = std::strlen(component->name());

    // Update component list
    const auto result = entries.emplace_back(component_header, component.value(),
                                             component_serializer.value());
    if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
  }

  return entries;
}

Expected<size_t> StdEntitySerializer::serializeComponents(
    const FixedVector<ComponentEntry, kMaxComponents>& entries, Endpoint* endpoint) {
  size_t size = 0;
  for (size_t i = 0; i < entries.size(); i++) {
    const auto& entry = entries[i];
    if (!entry) { return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE}; }
    const auto result = SerializeComponentHeader(entry->header, endpoint)
        .map([&](size_t component_header_size) { size += component_header_size; })
        .and_then([&]() {
          return endpoint->write(entry->component.name(), entry->header.name_size);
        })
        .and_then([&]() { size += entry->header.name_size; })
        .and_then([&]() {
          return entry->serializer->serializeComponent(entry->component, endpoint);
        })
        .map([&](size_t component_size) { size += component_size; });
    if (!result) { return ForwardError(result); }
  }
  return size;
}

Expected<void> StdEntitySerializer::deserializeComponents(
    size_t component_count, Entity entity, Endpoint* endpoint) {
  for (size_t i = 0; i < component_count; i++) {
    ComponentEntry entry;
    const auto result = DeserializeComponentHeader(endpoint)
        .assign_to(entry.header)
        .and_then([&]() { return findComponentSerializer(entry.header.tid); })
        .assign_to(entry.serializer)
        .and_then([&]() -> Expected<std::string> {
          try {
            std::string name(entry.header.name_size, '\0');
            return ExpectedOrError(endpoint->read(const_cast<char*>(name.data()), name.size()),
                                   name);
          } catch (const std::exception& exception) {
            GXF_LOG_ERROR("Failed to deserialize component name: %s", exception.what());
            return Unexpected{GXF_OUT_OF_MEMORY};
          }
        })
        .map([&](std::string name) { return entity.add(entry.header.tid, name.c_str()); })
        .assign_to(entry.component)
        .and_then([&]() {
          return entry.serializer->deserializeComponent(entry.component, endpoint);
        });
    if (!result) { return ForwardError(result); }
  }
  return Success;
}

Expected<Handle<ComponentSerializer>> StdEntitySerializer::findComponentSerializer(gxf_tid_t tid) {
  // Search cache for valid serializer
  const auto search = serializer_cache_.find(tid);
  if (search != serializer_cache_.end()) {
    return search->second;
  }

  // Search serializer list for valid serializer and cache result
  for (size_t i = 0; i < component_serializers_.get().size(); i++) {
    const auto component_serializer = component_serializers_.get()[i];
    if (!component_serializer) { return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE}; }
    if (component_serializer.value()->isSupported(tid)) {
      serializer_cache_[tid] = component_serializer.value();
      return component_serializer.value();
    }
  }

  return Unexpected{GXF_QUERY_NOT_FOUND};
}

}  // namespace gxf
}  // namespace nvidia
