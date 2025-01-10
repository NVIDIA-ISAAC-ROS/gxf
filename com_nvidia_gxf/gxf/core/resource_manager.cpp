/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/core/resource_manager.hpp"

namespace nvidia {
namespace gxf {

ResourceManager::ResourceManager(gxf_context_t context) {
  context_ = context;
}

// Static public interface impl
Expected<gxf_uid_t> ResourceManager::findEntityResourceByTypeName(gxf_context_t context,
                                       gxf_uid_t eid, const char* type_name,
                                       const char* target_resource_name) {
  const char* entity_name = nullptr;
  gxf_result_t result0 = GxfEntityGetName(context, eid, &entity_name);
  if (result0 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to obtain name of entity [eid: %05zu], %s", eid, GxfResultStr(result0));
  }

  gxf_uid_t resource_cids[kMaxComponents];
  uint64_t num_resource_cids = kMaxComponents;
  const gxf_result_t result1 = GxfEntityGroupFindResources(context, eid, &num_resource_cids,
                                                          resource_cids);
  if (result1 != GXF_SUCCESS) {
    GXF_LOG_ERROR(
      "Could not find all resource components from EntityGroup of entity %s (E%05zu)",
      entity_name, eid);
    return Unexpected{result1};
  }

  // find tid to the template type
  gxf_tid_t target_resource_tid;
  const auto result2 = GxfComponentTypeId(context, type_name, &target_resource_tid);
  if (result2 != GXF_SUCCESS) {
    GXF_LOG_WARNING("ResourceManager: Runtime cannot find tid of resource [type: %s]",
                    type_name);
    return Unexpected { result2 };
  }

  // iterate through all resource cid and find the matched one
  for (size_t i = 0; i < num_resource_cids; i++) {
    // get tid from cid
    gxf_uid_t resource_cid = resource_cids[i];
    gxf_tid_t resource_tid;
    const auto result3 = GxfComponentType(context, resource_cid, &resource_tid);
    if (result3 != GXF_SUCCESS) {
      return Unexpected { result3 };
    }
    // get name from cid
    const char* resource_name;
    if (target_resource_name != nullptr) {
      const auto result4 = GxfComponentName(context, resource_cid, &resource_name);
      if (result4 != GXF_SUCCESS) {
        return Unexpected { result4 };
      }
    }
    // check if resource_cid matches target by tid, AND name if provided
    bool match_condition = (resource_tid == target_resource_tid &&
      (target_resource_name == nullptr || std::strcmp(target_resource_name, resource_name) == 0));
    if (match_condition) {
      // return gxf_uid_t of the resource component
      GXF_LOG_DEBUG("ResourceManager find resource_cid [cid: %05zu, type: %s] "
                    "for entity [eid: %05zu, name: %s]",
                    resource_cid, type_name, eid, entity_name);
      return resource_cid;
    }
  }

  GXF_LOG_VERBOSE("ResourceManager cannot find Resource of "
                  "type: %s for entity [eid: %05zu, name: %s]",
                  type_name, eid, entity_name);
  return Unexpected { GXF_ENTITY_COMPONENT_NOT_FOUND };
}


// Static public interface impl
Expected<gxf_uid_t> ResourceManager::findComponentResourceByTypeName(gxf_context_t context,
                                     gxf_uid_t cid, const char* type_name,
                                     const char* target_resource_name) {
  const char* component_name;
  gxf_result_t result = GxfComponentName(context, cid, &component_name);
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("ResourceManager: Runtime cannot find name of component [cid: %05zu]", cid);
    return Unexpected { result };
  }
  gxf_uid_t eid;
  gxf_result_t result1 = GxfComponentEntity(context, cid, &eid);
  if (result1 != GXF_SUCCESS) {
    GXF_LOG_ERROR("ResourceManager: Runtime cannot find "
                  "eid of component [cid: %05zu, name: %s]", cid, component_name);
    return Unexpected { result1 };
  }
  auto maybe_cid = ResourceManager::findEntityResourceByTypeName(context,
                                    eid, type_name, target_resource_name);
  if (!maybe_cid) {
    GXF_LOG_VERBOSE("ResourceManager cannot find Resource of "
                  "type: %s for component [cid: %05zu, name: %s]",
                  type_name, cid, component_name);
  }
  return maybe_cid;
}

}  // namespace gxf
}  // namespace nvidia
