/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/core/gxf.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>  // NOLINT
#include <sstream>
#include <utility>

#include "common/assert.hpp"
#include "common/type_name.hpp"
#include "gxf/core/runtime.hpp"
#include "gxf/std/extension.hpp"
#include "yaml-cpp/yaml.h"

namespace nvidia {
namespace gxf {

gxf_context_t ToContext(Runtime* runtime) {
  return static_cast<gxf_context_t>(runtime);
}

Runtime* FromContext(gxf_context_t context) {
  return static_cast<Runtime*>(context);
}

// FIXME prototype code

}  // namespace gxf
}  // namespace nvidia

extern "C" {

bool isSuccessful(gxf_result_t result) {
  return result == GXF_SUCCESS ? true : false;
}

#define GXF_STR_HELP(X) \
  case X:               \
    return #X;

const char* GxfResultStr(gxf_result_t result) {
  switch (result) {
    GXF_STR_HELP(GXF_SUCCESS)
    GXF_STR_HELP(GXF_FAILURE)
    GXF_STR_HELP(GXF_NOT_IMPLEMENTED)
    GXF_STR_HELP(GXF_FILE_NOT_FOUND)
    GXF_STR_HELP(GXF_INVALID_ENUM)
    GXF_STR_HELP(GXF_NULL_POINTER)
    GXF_STR_HELP(GXF_UNINITIALIZED_VALUE)
    GXF_STR_HELP(GXF_ARGUMENT_NULL)
    GXF_STR_HELP(GXF_ARGUMENT_OUT_OF_RANGE)
    GXF_STR_HELP(GXF_ARGUMENT_INVALID)
    GXF_STR_HELP(GXF_OUT_OF_MEMORY)
    GXF_STR_HELP(GXF_MEMORY_INVALID_STORAGE_MODE)
    GXF_STR_HELP(GXF_CONTEXT_INVALID)
    GXF_STR_HELP(GXF_EXTENSION_NOT_FOUND)
    GXF_STR_HELP(GXF_EXTENSION_FILE_NOT_FOUND)
    GXF_STR_HELP(GXF_EXTENSION_NO_FACTORY)
    GXF_STR_HELP(GXF_FACTORY_TOO_MANY_COMPONENTS)
    GXF_STR_HELP(GXF_FACTORY_DUPLICATE_TID)
    GXF_STR_HELP(GXF_FACTORY_UNKNOWN_TID)
    GXF_STR_HELP(GXF_FACTORY_ABSTRACT_CLASS)
    GXF_STR_HELP(GXF_FACTORY_UNKNOWN_CLASS_NAME)
    GXF_STR_HELP(GXF_FACTORY_INVALID_INFO)
    GXF_STR_HELP(GXF_FACTORY_INCOMPATIBLE)
    GXF_STR_HELP(GXF_ENTITY_NOT_FOUND)
    GXF_STR_HELP(GXF_ENTITY_COMPONENT_NOT_FOUND)
    GXF_STR_HELP(GXF_ENTITY_COMPONENT_NAME_EXCEEDS_LIMIT)
    GXF_STR_HELP(GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION)
    GXF_STR_HELP(GXF_PARAMETER_NOT_FOUND)
    GXF_STR_HELP(GXF_PARAMETER_ALREADY_REGISTERED)
    GXF_STR_HELP(GXF_PARAMETER_INVALID_TYPE)
    GXF_STR_HELP(GXF_PARAMETER_OUT_OF_RANGE)
    GXF_STR_HELP(GXF_PARAMETER_NOT_INITIALIZED)
    GXF_STR_HELP(GXF_PARAMETER_CAN_NOT_MODIFY_CONSTANT)
    GXF_STR_HELP(GXF_PARAMETER_PARSER_ERROR)
    GXF_STR_HELP(GXF_PARAMETER_NOT_NUMERIC)
    GXF_STR_HELP(GXF_PARAMETER_MANDATORY_NOT_SET)
    GXF_STR_HELP(GXF_CONTRACT_INVALID_SEQUENCE)
    GXF_STR_HELP(GXF_CONTRACT_PARAMETER_NOT_SET)
    GXF_STR_HELP(GXF_CONTRACT_MESSAGE_NOT_AVAILABLE)
    GXF_STR_HELP(GXF_INVALID_LIFECYCLE_STAGE)
    GXF_STR_HELP(GXF_INVALID_EXECUTION_SEQUENCE)
    GXF_STR_HELP(GXF_REF_COUNT_NEGATIVE)
    GXF_STR_HELP(GXF_RESULT_ARRAY_TOO_SMALL)
    GXF_STR_HELP(GXF_INVALID_DATA_FORMAT)
    GXF_STR_HELP(GXF_EXCEEDING_PREALLOCATED_SIZE)
    GXF_STR_HELP(GXF_QUERY_NOT_ENOUGH_CAPACITY)
    GXF_STR_HELP(GXF_QUERY_NOT_APPLICABLE)
    GXF_STR_HELP(GXF_QUERY_NOT_FOUND)
    GXF_STR_HELP(GXF_NOT_FINISHED)
    GXF_STR_HELP(GXF_HTTP_GET_FAILURE)
    GXF_STR_HELP(GXF_HTTP_POST_FAILURE)
    GXF_STR_HELP(GXF_ENTITY_GROUP_NOT_FOUND)
    GXF_STR_HELP(GXF_RESOURCE_NOT_INITIALIZED)
    GXF_STR_HELP(GXF_RESOURCE_NOT_FOUND)
    GXF_STR_HELP(GXF_CONNECTION_BROKEN)
    GXF_STR_HELP(GXF_CONNECTION_ATTEMPTS_EXCEEDED)
    default:
      return "N/A";
  }
}

const char* GxfParameterTypeStr(gxf_parameter_type_t param_type) {
  switch (param_type) {
    GXF_STR_HELP(GXF_PARAMETER_TYPE_CUSTOM);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_HANDLE);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_STRING);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_INT64);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_UINT64);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_FLOAT64);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_BOOL);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_INT32);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_FILE);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_INT8);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_INT16);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_UINT8);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_UINT16);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_UINT32);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_FLOAT32);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_COMPLEX64);
    GXF_STR_HELP(GXF_PARAMETER_TYPE_COMPLEX128);
    default:
      return "N/A";
  }
}

const char* GxfParameterFlagTypeStr(gxf_parameter_flags_t_ flag_type) {
  switch (flag_type) {
    GXF_STR_HELP(GXF_PARAMETER_FLAGS_NONE);
    GXF_STR_HELP(GXF_PARAMETER_FLAGS_OPTIONAL);
    GXF_STR_HELP(GXF_PARAMETER_FLAGS_DYNAMIC);
    default:
      return "N/A";
  }
}

const char* GxfEntityStatusStr(gxf_entity_status_t status) {
  switch (status) {
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_NOT_STARTED, NotStarted)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_START_PENDING, StartPending)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_STARTED, Started)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_TICK_PENDING, TickPending)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_TICKING, Ticking)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_IDLE, Idle)
    GXF_ENUM_TO_STR(GXF_ENTITY_STATUS_STOP_PENDING, StopPending)
    default:
      return "N/A";
  }
}

const char* GxfEventStr(gxf_event_t event) {
  switch (event) {
    GXF_ENUM_TO_STR(GXF_EVENT_CUSTOM, Custom)
    GXF_ENUM_TO_STR(GXF_EVENT_EXTERNAL, External)
    GXF_ENUM_TO_STR(GXF_EVENT_MEMORY_FREE, MemoryFree)
    GXF_ENUM_TO_STR(GXF_EVENT_MESSAGE_SYNC, MessageSync)
    GXF_ENUM_TO_STR(GXF_EVENT_TIME_UPDATE, TimeUpdate)
    GXF_ENUM_TO_STR(GXF_EVENT_STATE_UPDATE, StateUpdate)
    default:
      return "N/A";
  }
}

#undef GXF_STR_HELP

gxf_result_t GxfContextCreate(gxf_context_t* context) {
  if (context == nullptr) return GXF_ARGUMENT_NULL;
  nvidia::gxf::Runtime* ptr = new nvidia::gxf::Runtime();
  if (ptr == nullptr) return GXF_OUT_OF_MEMORY;
  *context = nvidia::gxf::ToContext(ptr);
  return ptr->create();
}

gxf_result_t GxfContextCreateShared(gxf_context_t shared, gxf_context_t* context) {
  if (context == nullptr || shared == nullptr) return GXF_ARGUMENT_NULL;
  nvidia::gxf::Runtime* ptr = new nvidia::gxf::Runtime();
  if (ptr == nullptr) return GXF_OUT_OF_MEMORY;
  *context = nvidia::gxf::ToContext(ptr);
  return ptr->create(shared);
}

gxf_result_t GxfContextDestroy(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  nvidia::gxf::Runtime* pointer = nvidia::gxf::FromContext(context);
  const gxf_result_t code = pointer->destroy();
  if (code != GXF_SUCCESS) return code;
  delete pointer;
  return GXF_SUCCESS;
}

gxf_result_t GxfGetSharedContext(gxf_context_t context, gxf_context_t* shared) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGetSharedContext(shared);
}

gxf_result_t GxfRegisterComponent(gxf_context_t context, gxf_tid_t tid, const char* name,
                                  const char* base_name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfRegisterComponent(tid, name, base_name);
}

gxf_result_t GxfRegisterComponentInExtension(gxf_context_t context, gxf_tid_t component_tid,
                                             gxf_tid_t extension_tid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfRegisterComponentInExtension(component_tid,
                                                                            extension_tid);
}

gxf_result_t GxfLoadExtensions(gxf_context_t context, const GxfLoadExtensionsInfo* info) {
  if (context == nullptr) {
    return GXF_CONTEXT_INVALID;
  }
  if (info == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  return nvidia::gxf::FromContext(context)->GxfLoadExtensions(*info);
}

gxf_result_t GxfLoadExtensionFromPointer(gxf_context_t context, void* extension_ptr) {
  if (context == nullptr) {
    return GXF_CONTEXT_INVALID;
  }
  if (extension_ptr == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  nvidia::gxf::Extension* extension = reinterpret_cast<nvidia::gxf::Extension*>(extension_ptr);
  return nvidia::gxf::FromContext(context)->GxfLoadExtensionFromPointer(extension);
}

gxf_result_t GxfGraphLoadFile(gxf_context_t context, const char* filename,
             const char* parameters_override[], const uint32_t num_overrides) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphLoadFile(filename, parameters_override,
                                                             num_overrides);
}

gxf_result_t GxfGraphLoadFileExtended(gxf_context_t context, const char* filename,
             const char* entity_prefix,
             const char* parameters_override[], const uint32_t num_overrides,
             gxf_uid_t parent_eid, void* prerequisites) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphLoadFileExtended(filename, entity_prefix,
                                                                     parameters_override,
                                                                     num_overrides, parent_eid,
                                                                     prerequisites);
}

gxf_result_t GxfGraphSetRootPath(gxf_context_t context, const char* path) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphSetRootPath(path);
}

gxf_result_t GxfGraphParseString(gxf_context_t context, const char* filename,
             const char* parameters_override[], const uint32_t num_overrides) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphParseString(filename, parameters_override,
                                                                num_overrides);
}

gxf_result_t GxfGraphSaveToFile(gxf_context_t context, const char* filename) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphSaveToFile(filename);
}

gxf_result_t GxfEntityActivate(gxf_context_t context, gxf_uid_t eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityActivate(eid);
}

gxf_result_t GxfEntityDeactivate(gxf_context_t context, gxf_uid_t eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityDeactivate(eid);
}

gxf_result_t GxfEntityDestroy(gxf_context_t context, gxf_uid_t eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityDestroy(eid);
}

gxf_result_t GxfEntityFind(gxf_context_t context, const char* name, gxf_uid_t* eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityFind(name, eid);
}

gxf_result_t GxfEntityFindAll(gxf_context_t context, uint64_t* num_entities, gxf_uid_t* entities) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityFindAll(num_entities, entities);
}

gxf_result_t GxfEntityRefCountInc(gxf_context_t context, gxf_uid_t eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityRefCountInc(eid);
}

gxf_result_t GxfEntityRefCountDec(gxf_context_t context, gxf_uid_t eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityRefCountDec(eid);
}

gxf_result_t GxfEntityGetRefCount(gxf_context_t context, gxf_uid_t eid, int64_t* count) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGetRefCount(eid, count);
}

gxf_result_t GxfEntityGetStatus(gxf_context_t context, gxf_uid_t eid,
                              gxf_entity_status_t* entity_status) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGetStatus(eid, entity_status);
}

gxf_result_t GxfEntityGetName(gxf_context_t context, gxf_uid_t eid,
                              const char** entity_name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGetName(eid, entity_name);
}

gxf_result_t GxfEntityGetState(gxf_context_t context, gxf_uid_t eid,
                              entity_state_t* behavior_status) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGetState(eid, behavior_status);
}

gxf_result_t GxfEntityEventNotify(gxf_context_t context, gxf_uid_t eid) {
  if (context == nullptr) {
    return GXF_CONTEXT_INVALID;
  }
  return nvidia::gxf::FromContext(context)->GxfEntityNotifyEventType(eid, GXF_EVENT_EXTERNAL);
}

gxf_result_t GxfEntityNotifyEventType(gxf_context_t context, gxf_uid_t eid, gxf_event_t event) {
  if (context == nullptr) {
    return GXF_CONTEXT_INVALID;
  }
  return nvidia::gxf::FromContext(context)->GxfEntityNotifyEventType(eid, event);
}

gxf_result_t GxfComponentTypeId(gxf_context_t context, const char* name, gxf_tid_t* tid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentTypeId(name, tid);
}

gxf_result_t GxfComponentTypeName(gxf_context_t context, gxf_tid_t tid, const char** name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentTypeName(tid, name);
}

gxf_result_t GxfComponentTypeNameFromUID(gxf_context_t context, gxf_uid_t cid, const char** name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentTypeNameFromUID(cid, name);
}

gxf_result_t GxfComponentName(gxf_context_t context, gxf_uid_t cid, const char** name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentName(cid, name);
}

gxf_result_t GxfComponentEntity(gxf_context_t context, gxf_uid_t cid, gxf_uid_t* eid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentEntity(cid, eid);
}

gxf_result_t GxfEntityGetItemPtr(gxf_context_t context, gxf_uid_t eid, void** ptr) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  if (ptr == nullptr) return GXF_ARGUMENT_NULL;
  if (*ptr != nullptr) return GXF_ARGUMENT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGetItemPtr(eid, ptr);
}

gxf_result_t GxfComponentAdd(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name,
                             gxf_uid_t* cid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  void* tmp;
  return nvidia::gxf::FromContext(context)->GxfComponentAdd(eid, tid, name, cid, &tmp);
}

gxf_result_t GxfComponentAddAndGetPtr(gxf_context_t context, void* item_ptr, gxf_tid_t tid,
   const char* name, gxf_uid_t* out_cid, void ** comp_ptr) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  if ((comp_ptr == nullptr) || (item_ptr == nullptr)) { return GXF_ARGUMENT_NULL; }
  if (*comp_ptr != nullptr) {  return GXF_ARGUMENT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfComponentAddWithItem(item_ptr, tid, name, out_cid,
                                                            comp_ptr);
}

gxf_result_t GxfComponentRemoveWithUID(gxf_context_t context, gxf_uid_t cid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentRemove(cid);
}

gxf_result_t GxfComponentRemove(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid,
 const char * name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentRemove(eid, tid, name);
}

gxf_result_t GxfComponentAddToInterface(gxf_context_t context, gxf_uid_t eid,
                                        gxf_uid_t cid, const char* name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return  nvidia::gxf::FromContext(context)->GxfComponentAddToInterface(eid, cid, name);
}

gxf_result_t GxfComponentFind(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name,
                              int32_t* offset, gxf_uid_t* cid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentFind(eid, tid, name, offset, cid);
}

gxf_result_t GxfComponentFindAndGetPtr(gxf_context_t context, gxf_uid_t eid,  void* item_ptr,
                     gxf_tid_t tid, const char* name, int32_t* offset, gxf_uid_t* cid, void** ptr) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  if ((ptr == nullptr) || (item_ptr == nullptr)) { return GXF_ARGUMENT_NULL; }
  if (*ptr != nullptr) {  return GXF_ARGUMENT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfComponentFind(eid, item_ptr, tid, name,
   offset, cid, ptr);
}

gxf_result_t GxfComponentFindAll(gxf_context_t context, gxf_uid_t eid, uint64_t* num_cids,
                                 gxf_uid_t* cids) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentFindAll(eid, num_cids, cids);
}

gxf_result_t GxfEntityGroupFindResources(gxf_context_t context, gxf_uid_t eid,
                                         uint64_t* num_resource_cids, gxf_uid_t* resource_cids) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGroupFindResources(eid, num_resource_cids,
                                                                        resource_cids);
}

gxf_result_t GxfEntityGroupId(gxf_context_t context, gxf_uid_t eid, gxf_uid_t* gid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGroupId(eid, gid);
}

gxf_result_t GxfEntityGroupName(gxf_context_t context, gxf_uid_t eid, const char** name) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityGroupName(eid, name);
}

gxf_result_t GxfParameterSetInt8(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int8_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetInt8(uid, key, value);
}

gxf_result_t GxfParameterSetInt16(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int16_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetInt16(uid, key, value);
}

gxf_result_t GxfParameterSetInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int32_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetInt32(uid, key, value);
}

gxf_result_t GxfParameterSetInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int64_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetInt64(uid, key, value);
}

gxf_result_t GxfParameterSetUInt8(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  uint8_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetUInt8(uid, key, value);
}

gxf_result_t GxfParameterSetUInt16(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint16_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetUInt16(uid, key, value);
}

gxf_result_t GxfParameterSetUInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint32_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetUInt32(uid, key, value);
}

gxf_result_t GxfParameterSetUInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint64_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetUInt64(uid, key, value);
}

gxf_result_t GxfParameterSetFloat32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                    float value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetFloat32(uid, key, value);
}

gxf_result_t GxfParameterSetFloat64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                    double value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetFloat64(uid, key, value);
}

gxf_result_t GxfParameterSetStr(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetStr(uid, key, value);
}

gxf_result_t GxfParameterSetHandle(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   gxf_uid_t value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetHandle(uid, key, value);
}

gxf_result_t GxfParameterSetBool(gxf_context_t context, gxf_uid_t uid, const char* key,
                                 bool value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetBool(uid, key, value);
}

gxf_result_t GxfParameterSet1DStrVector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                        const char* value[], uint64_t length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet1DVectorString(uid, key, value, length);
}

gxf_result_t GxfParameterSet1DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double* value, uint64_t length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet1DVector<double>(uid, key, value,
                                                                            length);
}

gxf_result_t GxfParameterSet2DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double** value, uint64_t height, uint64_t width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet2DVector<double>(uid, key, value, height,
                                                                            width);
}

gxf_result_t GxfParameterSet1DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t* value, uint64_t length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet1DVector<int64_t>(uid, key, value,
                                                                             length);
}

gxf_result_t GxfParameterSet2DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t** value, uint64_t height, uint64_t width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet2DVector<int64_t>(uid, key, value,
                                                                             height, width);
}

gxf_result_t GxfParameterSet1DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t* value, uint64_t length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet1DVector<uint64_t>(uid, key, value,
                                                                              length);
}

gxf_result_t GxfParameterSet2DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t** value, uint64_t height, uint64_t width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet2DVector<uint64_t>(uid, key, value,
                                                                              height, width);
}

gxf_result_t GxfParameterSet1DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t* value, uint64_t length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet1DVector<int32_t>(uid, key, value,
                                                                             length);
}

gxf_result_t GxfParameterSet2DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t** value, uint64_t height, uint64_t width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSet2DVector<int32_t>(uid, key, value,
                                                                             height, width);
}

gxf_result_t GxfParameterSetFromYamlNode(gxf_context_t context, gxf_uid_t uid, const char* key,
                                         void* yaml_node, const char* prefix) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetFromYamlNode(uid, key, yaml_node,
                                                                        prefix);
}

gxf_result_t GxfParameterGetAsYamlNode(gxf_context_t context, gxf_uid_t uid, const char* key,
                                      void* yaml_node) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetAsYamlNode(uid, key, yaml_node);
}

gxf_result_t GxfParameterSetPath(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterSetPath(uid, key, value);
}

gxf_result_t GxfParameterGetFloat64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                    double* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetFloat64(uid, key, value);
}

gxf_result_t GxfParameterGetFloat32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                    float* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetFloat32(uid, key, value);
}

gxf_result_t GxfParameterGetInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int64_t* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetInt64(uid, key, value);
}

gxf_result_t GxfParameterGetUInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint64_t* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetUInt64(uid, key, value);
}

gxf_result_t GxfParameterGetUInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint32_t* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetUInt32(uid, key, value);
}

gxf_result_t GxfParameterGetUInt16(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint16_t* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetUInt16(uid, key, value);
}

gxf_result_t GxfParameterGetStr(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char** value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetStr(uid, key, value);
}

gxf_result_t GxfParameterGetPath(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char** value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetPath(uid, key, value);
}

gxf_result_t GxfParameterGetHandle(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   gxf_uid_t* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetHandle(uid, key, value);
}

gxf_result_t GxfParameterGetBool(gxf_context_t context, gxf_uid_t uid, const char* key,
                                 bool* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetBool(uid, key, value);
}

gxf_result_t GxfParameterGetInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int32_t* value) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGetInt32(uid, key, value);
}

gxf_result_t GxfParameterGet1DFloat64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                                const char* key, uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVectorInfo<double>(uid, key, length);
}

gxf_result_t GxfParameterGet2DFloat64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                                const char* key, uint64_t* height,
                                                uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVectorInfo<double>(uid, key, height,
                                                                                width);
}

gxf_result_t GxfParameterGet1DInt64VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVectorInfo<int64_t>(uid, key, length);
}

gxf_result_t GxfParameterGet2DInt64VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVectorInfo<int64_t>(uid, key, height,
                                                                                 width);
}

gxf_result_t GxfParameterGet1DUInt64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                               const char* key, uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVectorInfo<uint64_t>(uid, key, length);
}

gxf_result_t GxfParameterGet2DUInt64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                               const char* key, uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVectorInfo<uint64_t>(uid, key, height,
                                                                                  width);
}

gxf_result_t GxfParameterGet1DInt32VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVectorInfo<int32_t>(uid, key, length);
}

gxf_result_t GxfParameterGet2DInt32VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVectorInfo<int32_t>(uid, key, height,
                                                                                 width);
}

gxf_result_t GxfParameterGet1DStrVector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                        char* value[], uint64_t* count, uint64_t* min_length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DStrVector(uid, key, value,
                                                                            count, min_length);
}

gxf_result_t GxfParameterGet1DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double* value, uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVector<double>(uid, key, value,
                                                                            length);
}

gxf_result_t GxfParameterGet2DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double** value, uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVector<double>(uid, key, value, height,
                                                                            width);
}

gxf_result_t GxfParameterGet1DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t* value, uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVector<int64_t>(uid, key, value,
                                                                             length);
}

gxf_result_t GxfParameterGet2DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t** value, uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVector<int64_t>(uid, key, value,
                                                                             height, width);
}

gxf_result_t GxfParameterGet1DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t* value, uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVector<uint64_t>(uid, key, value,
                                                                              length);
}

gxf_result_t GxfParameterGet2DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t** value, uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVector<uint64_t>(uid, key, value,
                                                                              height, width);
}

gxf_result_t GxfParameterGet1DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t* value, uint64_t* length) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet1DVector<int32_t>(uid, key, value,
                                                                             length);
}

gxf_result_t GxfParameterGet2DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t** value, uint64_t* height, uint64_t* width) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterGet2DVector<int32_t>(uid, key, value,
                                                                             height, width);
}

gxf_result_t GxfComponentType(gxf_context_t context, gxf_uid_t cid, gxf_tid_t* tid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentType(cid, tid);
}

gxf_result_t GxfComponentPointer(gxf_context_t context, gxf_uid_t uid, gxf_tid_t tid,
                                 void** result) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentPointer(uid, tid, result);
}

gxf_result_t GxfComponentIsBase(gxf_context_t context, gxf_tid_t derived, gxf_tid_t base,
                                bool* result) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentIsBase(derived, base, result);
}

gxf_result_t GxfGraphActivate(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphActivate();
}

gxf_result_t GxfGraphDeactivate(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphDeactivate();
}

gxf_result_t GxfGraphRunAsync(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphRunAsync();
}

gxf_result_t GxfGraphInterrupt(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphInterrupt();
}

gxf_result_t GxfGraphWait(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphWait();
}

gxf_result_t GxfGraphRun(gxf_context_t context) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGraphRun();
}

gxf_result_t GxfRuntimeInfo(gxf_context_t context, gxf_runtime_info* info) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfRuntimeInfo(info);
}

gxf_result_t GxfExtensionInfo(gxf_context_t context, gxf_tid_t tid, gxf_extension_info_t* info) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfExtensionInfo(tid, info);
}

gxf_result_t GxfComponentInfo(gxf_context_t context, gxf_tid_t tid, gxf_component_info_t* info) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfComponentInfo(tid, info);
}

gxf_result_t GxfParameterInfo(gxf_context_t context, gxf_tid_t cid, const char* key,
                              gxf_parameter_info_t* info) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfParameterInfo(cid, key, info);
}

gxf_result_t GxfGetParameterInfo(gxf_context_t context, gxf_tid_t cid, const char* key,
                              gxf_parameter_info_t* info) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfGetParameterInfo(cid, key, info);
}

gxf_result_t GxfCreateEntityAndGetItem(gxf_context_t context, const GxfEntityCreateInfo* info,
                             gxf_uid_t* eid, void** item_ptr) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  if ((info == nullptr) || (eid == nullptr) || (item_ptr == nullptr)) { return GXF_ARGUMENT_NULL; }
  if (*item_ptr != nullptr) { return GXF_ARGUMENT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfCreateEntity(*info, *eid, item_ptr);
}

gxf_result_t GxfCreateEntity(gxf_context_t context, const GxfEntityCreateInfo* info,
                             gxf_uid_t* eid) {
  if (context == nullptr) {
    return GXF_CONTEXT_INVALID;
  }
  if (info == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (eid == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  return nvidia::gxf::FromContext(context)->GxfCreateEntity(*info, *eid);
}

gxf_result_t GxfLoadExtensionMetadataFiles(gxf_context_t context, const char* const* filenames,
                                   uint32_t count) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfLoadExtensionMetadataFiles(filenames, count);
}

gxf_result_t GxfSetSeverity(gxf_context_t context, gxf_severity_t severity) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfSetSeverity(severity);
}

gxf_result_t GxfGetSeverity(gxf_context_t context, gxf_severity_t* severity) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfGetSeverity(severity);
}

gxf_result_t GxfRedirectLog(gxf_context_t context, FILE* fp) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  return nvidia::gxf::FromContext(context)->GxfRedirectLog(fp);
}

gxf_result_t GxfCreateEntityGroup(gxf_context_t context, const char* name, gxf_uid_t* gid) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  if (gid == nullptr) { return GXF_ARGUMENT_NULL; }
  return nvidia::gxf::FromContext(context)->GxfCreateEntityGroup(name, gid);
}

gxf_result_t GxfUpdateEntityGroup(gxf_context_t context, gxf_uid_t gid, gxf_uid_t eid) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  if (gid == kNullUid) { return GXF_ARGUMENT_NULL; }
  if (eid == kNullUid) { return GXF_ARGUMENT_NULL; }
  return nvidia::gxf::FromContext(context)->GxfUpdateEntityGroup(gid, eid);
}

gxf_result_t GxfEntityIsValid(gxf_context_t context, gxf_uid_t eid, bool* valid) {
  if (context == nullptr) { return GXF_CONTEXT_INVALID; }
  if (valid == nullptr) { return GXF_ARGUMENT_NULL; }
  return nvidia::gxf::FromContext(context)->GxfEntityIsValid(eid, valid);
}

gxf_result_t GxfEntityResourceGetHandle(gxf_context_t context, gxf_uid_t eid,
               const char* type, const char* resource_key, gxf_uid_t* resource_cid) {
  if (context == nullptr) return GXF_CONTEXT_INVALID;
  return nvidia::gxf::FromContext(context)->GxfEntityResourceGetHandle(eid, type,
                                              resource_key, resource_cid);
}

}  // extern "C"
