/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CORE_RUNTIME_HPP_
#define NVIDIA_GXF_CORE_RUNTIME_HPP_

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>  // NOLINT
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gxf/core/gxf.h"

#include "common/assert.hpp"
#include "common/logger.hpp"
#include "common/type_name.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/parameter_registrar.hpp"
#include "gxf/core/parameter_storage.hpp"
#include "gxf/core/registrar.hpp"
#include "gxf/core/resource_manager.hpp"
#include "gxf/core/resource_registrar.hpp"
#include "gxf/core/type_registry.hpp"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/entity_warden.hpp"
#include "gxf/std/extension_loader.hpp"
#include "gxf/std/program.hpp"
#include "gxf/std/system.hpp"
#include "gxf/std/yaml_file_loader.hpp"
#include "yaml-cpp/yaml.h"

namespace nvidia {
namespace gxf {

class Runtime;

class SharedContext {
 public:
  gxf_context_t context();

  gxf_result_t create(gxf_context_t context);

  gxf_result_t initialize(Runtime* rt);

  gxf_result_t destroy();

  gxf_result_t removeComponentPointers(
      const FixedVector<gxf_uid_t, kMaxComponents>& cids);

  gxf_result_t removeSingleComponentPointer(gxf_uid_t& cid);

  gxf_result_t addComponent(gxf_uid_t cid, void* raw_pointer);

  gxf_result_t findComponentPointer(gxf_context_t context, gxf_uid_t uid, void** pointer);

  gxf_result_t loadExtensionImpl(const std::string& filename);

  gxf_result_t loadExtensionImpl(Extension& extension);

  gxf_uid_t getNextId();

 private:
  ExtensionLoader extension_loader_;

  EntityWarden warden_;

  TypeRegistry type_registry_;

  std::shared_ptr<ParameterStorage> parameters_;

  Registrar registrar_;

  // Stores information about parameter for query
  ParameterRegistrar parameter_registrar_;

  std::shared_ptr<ResourceRegistrar> resource_registrar_;

  std::shared_ptr<ResourceManager> resource_manager_;

  std::atomic<gxf_uid_t> next_id_{kNullUid + 1};

  std::unordered_map<gxf_uid_t, void*> objects_;

  std::mutex load_extension_mutex_;

  std::shared_timed_mutex global_object_mutex_;

 private:
  gxf_result_t createDefaultEntityGroup();
};

class Runtime {
 public:
  gxf_context_t context();

  gxf_result_t create();

  gxf_result_t create(gxf_context_t shared);

  gxf_result_t destroy();

  gxf_result_t GxfRegisterComponent(gxf_tid_t tid, const char* name, const char* base);

  gxf_result_t GxfRegisterComponentInExtension(const gxf_tid_t& component_tid,
                                               const gxf_tid_t& extension_tid);

  gxf_result_t GxfGetSharedContext(void** shared);

  // Gets version information about Runtime and list of loaded Extensions.
  gxf_result_t GxfRuntimeInfo(gxf_runtime_info* info);

  // Gets description of loaded extension and list of components it provides
  gxf_result_t GxfExtensionInfo(const gxf_tid_t eid, gxf_extension_info_t* info);

  // Gets description of component and list of parameter. List parameter is only
  // available if the component is already instantiated.
  gxf_result_t GxfComponentInfo(const gxf_tid_t tid, gxf_component_info_t* info);

  gxf_result_t GxfGetParameterInfo(const gxf_tid_t cid, const char* key,
               gxf_parameter_info_t* info);
  gxf_result_t GxfParameterInfo(const gxf_tid_t cid, const char* key, gxf_parameter_info_t* info);

  gxf_result_t loadExtensionImpl(const std::string& filename);

  gxf_result_t loadExtensionImpl(Extension& extension);

  gxf_result_t GxfLoadExtensions(const GxfLoadExtensionsInfo& info);

  gxf_result_t GxfLoadExtensionFromPointer(Extension* extension);

  gxf_result_t GxfGraphLoadFile(const char* filename,
                                const char* params_override[],
                                const uint32_t num_overrides);

  gxf_result_t GxfGraphLoadFileExtended(const char* filename, const char* entity_prefix,
                                        const char* params_override[], const uint32_t num_overrides,
                                        gxf_uid_t parent_eid = kNullUid,
                                        void* prerequisites = nullptr);

  gxf_result_t GxfGraphSetRootPath(const char* path);

  gxf_result_t GxfGraphParseString(const char* text,
                                   const char* params_override[],
                                   const uint32_t num_overrides);

  gxf_result_t GxfGraphSaveToFile(const char* filename);

  gxf_result_t GxfCreateEntity(const GxfEntityCreateInfo& info,
                               gxf_uid_t& eid,
                               void** item_ptr = nullptr);

  gxf_result_t GxfCreateEntityGroup(const char* name, gxf_uid_t* gid);

  gxf_result_t GxfUpdateEntityGroup(gxf_uid_t gid, gxf_uid_t eid);

  gxf_result_t GxfEntityIsValid(gxf_uid_t eid, bool* valid);

  gxf_result_t GxfEntityActivate(gxf_uid_t eid);

  gxf_result_t GxfEntityDeactivate(gxf_uid_t eid);

  gxf_result_t GxfEntityDestroyImpl(gxf_uid_t eid);

  gxf_result_t GxfEntityDestroy(gxf_uid_t eid);

  gxf_result_t GxfEntityFind(const char* name, gxf_uid_t* eid);

  gxf_result_t GxfEntityFindAll(uint64_t* num_entities,
                                gxf_uid_t* entities);

  gxf_result_t GxfEntityRefCountInc(gxf_uid_t eid);

  gxf_result_t GxfEntityRefCountDec(gxf_uid_t eid);

  gxf_result_t GxfEntityGetRefCount(gxf_uid_t eid, int64_t* count) const;

  gxf_result_t GxfEntityGetStatus(gxf_uid_t eid, gxf_entity_status_t* entity_status);

  gxf_result_t GxfEntityGetName(gxf_uid_t eid, const char** entity_name);

  gxf_result_t GxfEntityGetState(gxf_uid_t eid, entity_state_t* behavior_status);

  gxf_result_t GxfEntityNotifyEventType(gxf_uid_t eid, gxf_event_t event);

  gxf_result_t GxfComponentTypeName(gxf_tid_t tid, const char** name);

  gxf_result_t GxfComponentTypeNameFromUID(gxf_uid_t cid, const char** name);

  gxf_result_t GxfComponentTypeId(const char* name, gxf_tid_t* tid);

  gxf_result_t GxfComponentName(gxf_uid_t cid, const char** name);

  gxf_result_t GxfComponentEntity(gxf_uid_t cid, gxf_uid_t* eid);

  gxf_result_t GxfEntityGetItemPtr(gxf_uid_t eid, void** ptr);

  gxf_result_t GxfComponentAddWithItem(void* item_ptr, gxf_tid_t tid, const char* name,
                               gxf_uid_t* out_cid, void** comp_ptr);

  gxf_result_t GxfComponentAdd(gxf_uid_t eid, gxf_tid_t tid, const char* name, gxf_uid_t* out_cid,
                               void** comp_ptr);

  gxf_result_t GxfComponentRemove(gxf_uid_t cid);

  gxf_result_t GxfComponentRemove(gxf_uid_t eid, gxf_tid_t tid, const char * name);

  gxf_result_t GxfComponentAddToInterface(gxf_uid_t eid, gxf_uid_t cid,
                                          const char* name);

  gxf_result_t GxfComponentFind(gxf_uid_t eid, gxf_tid_t tid, const char* name, int32_t* offset,
                                gxf_uid_t* cid);

  gxf_result_t GxfComponentFind(gxf_uid_t eid, void* item_ptr, gxf_tid_t tid, const char* name,
                                int32_t* offset, gxf_uid_t* cid, void** ptr);

  gxf_result_t GxfComponentFindAll(gxf_uid_t eid, uint64_t* num_cids, gxf_uid_t* cids);

  gxf_result_t GxfComponentIsBase(gxf_tid_t derived, gxf_tid_t base, bool* result);

  gxf_result_t GxfEntityGroupFindResources(gxf_uid_t eid, uint64_t* num_resource_cids,
                                           gxf_uid_t* resource_cids);

  gxf_result_t GxfEntityGroupId(gxf_uid_t eid, gxf_uid_t* gid);

  gxf_result_t GxfEntityGroupName(gxf_uid_t eid, const char** name);

  gxf_result_t GxfEntityResourceGetHandle(gxf_uid_t eid, const char* type,
                        const char* resource_key, gxf_uid_t* resource_cid);

  gxf_result_t GxfParameterSetInt8(gxf_uid_t uid, const char* key, int8_t value);

  gxf_result_t GxfParameterSetInt16(gxf_uid_t uid, const char* key, int16_t value);

  gxf_result_t GxfParameterSetInt32(gxf_uid_t uid, const char* key, int32_t value);

  gxf_result_t GxfParameterSetInt64(gxf_uid_t uid, const char* key, int64_t value);

  gxf_result_t GxfParameterSetUInt8(gxf_uid_t uid, const char* key, uint8_t value);

  gxf_result_t GxfParameterSetUInt16(gxf_uid_t uid, const char* key, uint16_t value);

  gxf_result_t GxfParameterSetUInt32(gxf_uid_t uid, const char* key, uint32_t value);

  gxf_result_t GxfParameterSetUInt64(gxf_uid_t uid, const char* key, uint64_t value);

  gxf_result_t GxfParameterSetFloat32(gxf_uid_t uid, const char* key, float value);

  gxf_result_t GxfParameterSetFloat64(gxf_uid_t uid, const char* key, double value);

  gxf_result_t GxfParameterSetStr(gxf_uid_t uid, const char* key, const char* value);

  gxf_result_t GxfParameterSetPath(gxf_uid_t uid, const char* key, const char* value);

  gxf_result_t GxfParameterSetHandle(gxf_uid_t uid, const char* key, gxf_uid_t value);

  gxf_result_t GxfParameterSetBool(gxf_uid_t uid, const char* key, bool value);

  gxf_result_t GxfParameterSet1DVectorString(gxf_uid_t uid, const char* key, const char* value[],
                                             uint64_t length);

  template <typename T>
  gxf_result_t GxfParameterSet1DVector(gxf_uid_t uid, const char* key, T* value, uint64_t length) {
    GXF_LOG_VERBOSE("[C%05zu] PROPERTY SET: '%s'", uid, key);
    if (length && !value) { return GXF_ARGUMENT_NULL; }  // value null when non-zero length
    std::vector<T> value_(length);
    if (length) { std::memcpy(value_.data(), value, length * sizeof(T)); }
    return ToResultCode(parameters_->set<std::vector<T>>(uid, key, value_));
  }

  template <typename T>
  gxf_result_t GxfParameterSet2DVector(gxf_uid_t uid, const char* key, T** value, uint64_t height,
                                       uint64_t width) {
    GXF_LOG_VERBOSE("[C%05zu] PROPERTY SET: '%s'", uid, key);
    if (height && width && !value) {
      return GXF_ARGUMENT_NULL;  // value null when non-zero height & width
    }
    std::vector<std::vector<T>> value_;
    for (uint i = 0; i < height; i++) {
      std::vector<T> inner(width);
      if (width) {  // if width zero then value[i] could be null
        inner.assign(value[i], value[i] + width);
      }
      value_.push_back(inner);
    }
    return ToResultCode(parameters_->set<std::vector<std::vector<T>>>(uid, key, value_));
  }

  gxf_result_t GxfParameterSetFromYamlNode(gxf_uid_t uid, const char* key, void* yaml_node,
                                           const char* prefix);

  gxf_result_t GxfParameterGetAsYamlNode(gxf_uid_t uid, const char* key, void* yaml_node);

  gxf_result_t GxfParameterGetFloat64(gxf_uid_t uid, const char* key, double* value);

  gxf_result_t GxfParameterGetFloat32(gxf_uid_t uid, const char* key, float* value);

  gxf_result_t GxfParameterGetInt64(gxf_uid_t uid, const char* key, int64_t* value);

  gxf_result_t GxfParameterGetUInt64(gxf_uid_t uid, const char* key, uint64_t* value);

  gxf_result_t GxfParameterGetUInt32(gxf_uid_t uid, const char* key, uint32_t* value);

  gxf_result_t GxfParameterGetUInt16(gxf_uid_t uid, const char* key, uint16_t* value);

  template <typename T>
  gxf_result_t GxfParameterGet1DVectorInfo(gxf_uid_t uid, const char* key, uint64_t* length) {
    GXF_LOG_VERBOSE("[C%05zu] PROPERTY GET: '%s'", uid, key);
    if (!length) { return GXF_ARGUMENT_NULL; }
    const Expected<std::vector<T>> result = parameters_->get<std::vector<T>>(uid, key);
    if (result) {
      *length = result.value().size();
      return GXF_SUCCESS;
    } else {
      return result.error();
    }
  }

  template <typename T>
  gxf_result_t GxfParameterGet2DVectorInfo(gxf_uid_t uid, const char* key, uint64_t* height,
                                           uint64_t* width) {
    GXF_LOG_VERBOSE("[C%05zu] PROPERTY GET: '%s'", uid, key);
    if (!height || !width) { return GXF_ARGUMENT_NULL; }
    const Expected<std::vector<std::vector<T>>> result =
        parameters_->get<std::vector<std::vector<T>>>(uid, key);
    if (result) {
      *height = result.value().size();
      *width = result.value()[0].size();
      return GXF_SUCCESS;
    } else {
      return result.error();
    }
  }

  gxf_result_t GxfParameterGet1DStrVector(gxf_uid_t uid, const char* key, char* value[],
                                          uint64_t* count, uint64_t* min_length);

  template <typename T>
  gxf_result_t GxfParameterGet1DVector(gxf_uid_t uid, const char* key, T* value, uint64_t* length) {
    GXF_LOG_VERBOSE("[C%05zu] PROPERTY GET: '%s'", uid, key);
    const Expected<std::vector<T>> result = parameters_->get<std::vector<T>>(uid, key);
    if (!length) return GXF_ARGUMENT_NULL;
    if (result) {
      uint64_t result_length = result.value().size();
      if (result_length) {               // non-zero result length
        if (result_length <= *length) {  // enough capacity
          *length = result_length;
          if (value) {  // valid pointer
            std::memcpy(value, result.value().data(), result_length * sizeof(T));
            return GXF_SUCCESS;
          } else {  // null pointer
            GXF_LOG_ERROR("value is null");
            return GXF_ARGUMENT_NULL;
          }
        } else {  // not enough capacity
          *length = result_length;
          return GXF_QUERY_NOT_ENOUGH_CAPACITY;
        }
      } else {  // zero result length
        *length = result_length;
        return GXF_SUCCESS;
      }
    } else {
      return result.error();
    }
  }

  template <typename T>
  gxf_result_t GxfParameterGet2DVector(gxf_uid_t uid, const char* key, T** value, uint64_t* height,
                                       uint64_t* width) {
    GXF_LOG_VERBOSE("[C%05zu] PROPERTY GET: '%s'", uid, key);
    const Expected<std::vector<std::vector<T>>> result =
        parameters_->get<std::vector<std::vector<T>>>(uid, key);
    if (!height || !width) return GXF_ARGUMENT_NULL;
    if (result) {
      uint64_t result_height = result.value().size();
      uint64_t result_width = result.value()[0].size();
      if (result_height && result_width) {                           // copy required
        if (value) {                                                 // valid pointer
          if (result_height <= *height && result_width <= *width) {  // enough capacity
            for (uint i = 0; i < result_height; i++) {
              std::memcpy(value[i], result.value()[i].data(), result_width * sizeof(T));
            }
            *height = result_height;
            *width = result_width;
            return GXF_SUCCESS;
          } else {  // not enough capacity
            *height = result_height;
            *width = result_width;
            return GXF_QUERY_NOT_ENOUGH_CAPACITY;
          }
        } else {  // null pointer
          GXF_LOG_ERROR("value is null");
          *height = result_height;
          *width = result_width;
          return GXF_ARGUMENT_NULL;
        }
      } else {  // copy not required
        *height = result_height;
        *width = result_width;
        return GXF_SUCCESS;
      }
    } else {
      return result.error();
    }
  }

  gxf_result_t GxfParameterInt64Add(gxf_uid_t uid, const char* key, int64_t delta, int64_t* value);

  gxf_result_t GxfParameterGetStr(gxf_uid_t uid, const char* key, const char** value);

  gxf_result_t GxfParameterGetPath(gxf_uid_t uid, const char* key, const char** value);

  gxf_result_t GxfParameterGetHandle(gxf_uid_t uid, const char* key, gxf_uid_t* value);

  gxf_result_t GxfComponentType(gxf_uid_t cid, gxf_tid_t* tid);

  gxf_result_t GxfParameterGetBool(gxf_uid_t uid, const char* key, bool* value);

  gxf_result_t GxfParameterGetInt32(gxf_uid_t uid, const char* key, int32_t* value);

  gxf_result_t GxfComponentPointer(gxf_uid_t uid, gxf_tid_t tid, void** pointer);

  gxf_result_t GxfGraphActivate();

  gxf_result_t GxfGraphRunAsync();

  gxf_result_t GxfGraphInterrupt();

  gxf_result_t GxfGraphWait();

  gxf_result_t GxfGraphDeactivate();

  gxf_result_t GxfGraphRun();

  gxf_result_t GxfLoadExtensionMetadataFiles(const char* const* filenames, uint32_t count);

  gxf_result_t GxfSetSeverity(gxf_severity_t severity);

  gxf_result_t GxfGetSeverity(gxf_severity_t* severity);

  gxf_result_t GxfRedirectLog(FILE* fp);

  gxf_result_t GxfSetExtensionLoader(ExtensionLoader* extension_loader);

  gxf_result_t GxfSetEntityWarden(EntityWarden* warden);

  gxf_result_t GxfSetTypeRegistry(TypeRegistry* type_registry);

  gxf_result_t GxfSetParameterStorage(std::shared_ptr<ParameterStorage> parameters);

  gxf_result_t GxfSetRegistrar(Registrar* registrar);

  gxf_result_t GxfSetParameterRegistrar(ParameterRegistrar* parameter_registrar);

  gxf_result_t GxfSetResourceRegistrar(std::shared_ptr<ResourceRegistrar> resource_registrar);

  gxf_result_t GxfSetResourceManager(std::shared_ptr<ResourceManager> resource_manager);

  gxf_result_t GxfSetDefaultEntityGroupId(gxf_uid_t* default_gid);

 private:
  gxf_result_t SearchLdLibraryPath(const std::string& filename);

  gxf_result_t GxfGraphLoadFileInternal(
      const char* filename, const char* entity_prefix, const char* params_override[],
      const uint32_t num_overrides, gxf_uid_t parent_eid = kNullUid,
      const YAML::Node& prerequisites = YAML::Node(YAML::NodeType::Null));

  gxf_tid_t component_tid_;
  gxf_tid_t sys_tid_;

  SharedContext* shared_context_;

  ExtensionLoader* extension_loader_{nullptr};

  EntityWarden* warden_{nullptr};

  TypeRegistry* type_registry_{nullptr};

  std::shared_ptr<ParameterStorage> parameters_;

  Registrar* registrar_{nullptr};

  // Stores information about parameter for query
  ParameterRegistrar* parameter_registrar_;

  std::shared_ptr<ResourceRegistrar> resource_registrar_;

  // have to stick to shared_ptr to avoid nullptr dereferencing risk,
  // as Codelet's Resource try_get() are lazy call
  std::shared_ptr<ResourceManager> resource_manager_;

  Program program_;

  EntityExecutor entity_executor_;

  // Current version of gxf core runtime
  const std::string gxf_core_version_{kGxfCoreVersion};

  std::shared_timed_mutex mutex_;

  // The root path for graph loading
  std::string graph_path_;

  bool shared_context_owner_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CORE_RUNTIME_HPP_
