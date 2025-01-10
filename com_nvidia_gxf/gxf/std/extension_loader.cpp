/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/extension_loader.hpp"

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "dlfcn.h"

#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/component_factory.hpp"

namespace nvidia {
namespace gxf {

namespace {
constexpr const char* kGxfExtensionFactoryName = "GxfExtensionFactory";
using GxfExtensionFactory = gxf_result_t(void**);
}  // namespace

gxf_result_t ExtensionLoader::allocate_abi(gxf_tid_t tid, void** out_pointer) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  if (out_pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  // Find the extension factory which can create this component.
  const auto it = component_factory_.find(tid);
  if (it == component_factory_.end()) {
    return GXF_FACTORY_UNKNOWN_TID;
  }

  // Create the component with that factory.
  const auto result = it->second->allocate(tid);
  if (!result) {
    return result.error();
  }
  *out_pointer = result.value();
  return GXF_SUCCESS;
}

gxf_result_t ExtensionLoader::deallocate_abi(gxf_tid_t tid, void* pointer) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);

  // Find the extension factory which can create this component.
  const auto it = component_factory_.find(tid);
  if (it == component_factory_.end()) {
    return GXF_FACTORY_UNKNOWN_TID;
  }

  // Create the component with that factory.
  const auto result = it->second->deallocate(tid, pointer);
  if (!result) {
    return result.error();
  }
  return GXF_SUCCESS;
}

Expected<void> ExtensionLoader::initialize(gxf_context_t context) {
  context_ = context;
  auto result = factories_.reserve(kMaxExtensions);
  if (!result) {
    GXF_LOG_WARNING("Memory allocation failed");
    return Unexpected{GXF_OUT_OF_MEMORY};
  }
  return Success;
}

static bool isCompatible(const std::string& ver1, const std::string& ver2) {
  std::vector<std::string> ver1Tokens, ver2Tokens;
  std::regex re("\\d+");

  std::sregex_token_iterator begin1(ver1.begin(), ver1.end(), re);
  std::sregex_token_iterator begin2(ver2.begin(), ver2.end(), re);
  std::sregex_token_iterator end;

  // Split version string into major, minor and build tokens
  std::copy(begin1, end, std::back_inserter(ver1Tokens));
  std::copy(begin2, end, std::back_inserter(ver2Tokens));

  bool compatible = true;

  // Check number of tokens in version numbers
  compatible &= (ver1Tokens.size() == 3);
  compatible &= (ver2Tokens.size() == 3);

  // Major and minor version check
  compatible &= (ver1Tokens[0] == ver2Tokens[0]);
  compatible &= (ver1Tokens[1] == ver2Tokens[1]);

  return compatible;
}

Expected<void> ExtensionLoader::load(const char* filename) {
  if (filename == nullptr) {
    GXF_LOG_ERROR("Extension filename is null");
    return Unexpected{GXF_NULL_POINTER};
  }

  void* handle = dlopen(filename, RTLD_LAZY);
  if (handle == nullptr) {
    GXF_LOG_ERROR("Failed to load extension %s Error: %s", filename, dlerror());
    return Unexpected{GXF_EXTENSION_FILE_NOT_FOUND};
  }

  void* function_pointer = dlsym(handle, kGxfExtensionFactoryName);
  if (function_pointer == nullptr) {
    GXF_LOG_ERROR("%s", dlerror());
    dlclose(handle);
    return Unexpected{GXF_EXTENSION_NO_FACTORY};
  }
  const auto factory_f = reinterpret_cast<GxfExtensionFactory*>(function_pointer);

  // Get the extension
  Extension* extension;
  {
    void* result;
    const gxf_result_t code = factory_f(&result);
    if (code != GXF_SUCCESS) return Unexpected{code};
    extension = static_cast<Extension*>(result);
    // FIXME How do we know we got a valid pointer?
  }

  const auto result = load(extension, handle);
  if (!result) {
    GXF_LOG_ERROR("Failed to load extension %s", filename);
  }

  return result;
}

Expected<void> ExtensionLoader::registerRuntimeComponent(const gxf_tid_t& component_tid,
                                                         const gxf_tid_t& ext_id) {
  const auto ext_it = extension_factory_.find(ext_id);
  if (ext_it == extension_factory_.end()) {
    GXF_LOG_ERROR("Extension not found. Have you loaded it ?");
    return Unexpected{GXF_EXTENSION_NOT_FOUND};
  }

  const auto extension = ext_it->second;
  if (!extension->hasComponent(component_tid)) {
    GXF_LOG_ERROR("Component not found. Have you loaded it in the extension ?");
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }

  const auto it = component_factory_.find(component_tid);
  if (it != component_factory_.end()) {
    GXF_LOG_ERROR("Duplicated component TID. TID: %016lx%016lx",
                  component_tid.hash1, component_tid.hash2);
    gxf_component_info_t info;
    const auto result = extension->getComponentInfo(component_tid, &info);
    if (result) {
      GXF_LOG_ERROR("Component name: %s", info.base_name);
    } else {
      GXF_LOG_ERROR("Component name: (error)");
    }
    return Unexpected{GXF_FACTORY_DUPLICATE_TID};
  }

  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  component_factory_[component_tid] = extension;
  return Success;
}


Expected<void> ExtensionLoader::load(Extension* extension, void* handle) {
  {
    auto result = extension->checkInfo();
    if (!result) { return ForwardError(result);}

    gxf_extension_info_t info;
    info.num_components = 0;
    result = extension->getInfo(&info);
    if (!result) { return ForwardError(result);}

    GXF_LOG_VERBOSE("Loading extension: %s", info.name);

    // Check if the gxf core version used to compile the extension is compatible
    // with the current runtime version
    if (!isCompatible(info.runtime_version, kGxfCoreVersion)) {
      GXF_LOG_ERROR("Runtime version mismatch. The extension was compiled with gxf core version %s "
                    "Current runtime version %s", info.runtime_version, kGxfCoreVersion);
      return Unexpected{GXF_FACTORY_INCOMPATIBLE};
    }
  }

  // Get all TIDs in this extension
  {
    FixedVector<gxf_tid_t, kMaxComponents> tids;
    auto maybe = tids.resize(tids.capacity());
    if (!maybe) {
      GXF_LOG_WARNING("Failed to resize vector");
      return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
    }

    size_t count = tids.size();
    const Expected<void> result = extension->getComponentTypes(tids.data(), &count);
    if (!result) { return ForwardError(result); }

    maybe = tids.resize(count);
    if (!maybe) {
      GXF_LOG_WARNING("Failed to resize vector");
      return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
    }

    for (size_t i = 0; i < tids.size(); i++) {
      const gxf_tid_t tid = tids.at(i).value();
      const auto it = component_factory_.find(tid);
      if (it != component_factory_.end()) {
        GXF_LOG_ERROR("Duplicated component TID. TID: %016lx%016lx", tid.hash1, tid.hash2);
        gxf_component_info_t info;
        const auto result = extension->getComponentInfo(tid, &info);
        if (result) {
          GXF_LOG_ERROR("Component name: %s", info.base_name);
        } else {
          GXF_LOG_ERROR("Component name: (error)");
        }
        return Unexpected{GXF_FACTORY_DUPLICATE_TID};
      }
      component_factory_[tid] = extension;
    }
  }

  // Register components in the extension
  const auto registration_result = extension->registerComponents(context_);
  if (!registration_result) { return ForwardError(registration_result); }

  // Add the extension
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  auto maybe = factories_.push_back(extension);
  if (!maybe) {
    GXF_LOG_WARNING("Exceeding maximum number of extensions");
    return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
  }

  // Finds extension metadata
  gxf_extension_info_t e_info;
  e_info.num_components = 0;
  auto result = extension->getInfo(&e_info);
  if (result) {
    extension_factory_[e_info.id] = extension;
  }

  // Add the handle
  if (handle) {
    handles_.insert(handle);
  }

  return Success;
}

Expected<void> ExtensionLoader::getComponentTypes(gxf_tid_t* pointer, size_t* size) {
  if (pointer == nullptr) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  if (size == nullptr) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  if (*size < component_factory_.size()) {
    return Unexpected{GXF_RESULT_ARRAY_TOO_SMALL};
  }

  uint64_t index = 0;
  for (const auto& kvp : component_factory_) {
    pointer[index] = kvp.first;
    index++;
  }
  *size = component_factory_.size();

  return Success;
}

Expected<void> ExtensionLoader::unloadAll() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  extension_factory_.clear();
  component_factory_.clear();
  // FIXME
  // for (void* handle : handles_) {
  //   dlclose(handle);
  // }
  handles_.clear();
  factories_.clear();

  return Success;
}

Expected<void> ExtensionLoader::getExtensions(uint64_t* extension_count, gxf_tid_t* extensions) {
  if (extension_count == nullptr) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }
  const uint64_t array_size = *extension_count;
  const uint64_t ext_num = static_cast<int64_t>(factories_.size());
  *extension_count = ext_num;
  if (array_size < ext_num) {
    return Unexpected{GXF_RESULT_ARRAY_TOO_SMALL};
  }
  if (*extension_count == 0) {
    return Success;
  }

  uint64_t factories_count = factories_.size();
  for (uint64_t i = 0; i < factories_count; ++i) {
    gxf_extension_info_t info;
    info.num_components = 0;
    auto result = factories_.at(i).value()->getInfo(&info);
    if (!result) { return ForwardError(result); }
    extensions[i] = info.id;
  }

  return Success;
}

Expected<void> ExtensionLoader::getExtensionInfo(gxf_tid_t eid, gxf_extension_info_t* info) {
  auto it = extension_factory_.find(eid);
  if (it == extension_factory_.end()) {
    return Unexpected{GXF_EXTENSION_NOT_FOUND};
  }
  return it->second->getInfo(info);
}

Expected<void> ExtensionLoader::getComponentInfo(const gxf_tid_t tid, gxf_component_info_t* info) {
  const auto it = component_factory_.find(tid);
  if (it == component_factory_.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }
  auto result = it->second->getComponentInfo(tid, info);
  if (!result) { return ForwardError(result); }

  return Success;
}

Expected<void> ExtensionLoader::getParameterInfo(const gxf_tid_t cid, const char* key,
               gxf_parameter_info_t* info) {
  const auto it = component_factory_.find(cid);
  if (it == component_factory_.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }
  auto result = it->second->getParameterInfo(context_, cid, key, info);
  if (!result) { return ForwardError(result); }

  return Success;
}


}  // namespace gxf
}  // namespace nvidia
