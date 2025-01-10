/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gxf/app/extension_manager.hpp"

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

Expected<void> ExtensionManager::load(const char* filename) {
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
  if (!result) { GXF_LOG_ERROR("Failed to load extension %s", filename); }

  return result;
}

Expected<void> ExtensionManager::load(Extension* extension, void* handle) {
  {
    auto result = extension->checkInfo();
    if (!result) { return ForwardError(result); }

    gxf_extension_info_t info;
    info.num_components = 0;
    result = extension->getInfo(&info);
    if (!result) { return ForwardError(result); }

    GXF_LOG_VERBOSE("Loading extension: %s", info.name);

    // Check if the gxf core version used to compile the extension is compatible
    // with the current runtime version
    if (!isCompatible(info.runtime_version, kGxfCoreVersion)) {
      GXF_LOG_ERROR(
          "Runtime version mismatch. The extension was compiled with gxf core version %s "
          "Current runtime version %s",
          info.runtime_version, kGxfCoreVersion);
      return Unexpected{GXF_FACTORY_INCOMPATIBLE};
    }
  }

  // Get all TIDs in this extension
  {
    FixedVector<gxf_tid_t> tids;
    tids.reserve(kMaxComponents);
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

  // Add the extension
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  factories_.push_back(extension);

  // Finds extension metadata
  gxf_extension_info_t e_info;
  e_info.num_components = 0;
  auto result = extension->getInfo(&e_info);
  if (result) { extension_factory_[e_info.id] = extension; }

  // Add the handle
  if (handle) { handles_.insert(handle); }

  return Success;
}

Expected<void> ExtensionManager::loadManifest(const char* filename) {
  try {
    const auto node = YAML::LoadFile(filename);
    for (const auto& entry : node["extensions"]) {
      auto file = entry.as<std::string>();
      auto result = load(file.c_str());
      if (!result) { return result; }
    }
  } catch (std::exception& x) {
    GXF_LOG_VERBOSE("Error loading manifest '%s': %s", filename, x.what());
    return Unexpected{};
  }

  return Success;
}

Expected<void> ExtensionManager::registerExtensions(gxf_context_t context) {
  for (const auto& extension : factories_) {
    gxf_result_t result = GxfLoadExtensionFromPointer(context, extension);
    if (result != GXF_SUCCESS) { return Unexpected{result}; }
  }
  return Success;
}

Expected<void> ExtensionManager::unloadAll() {
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

}  // namespace gxf
}  // namespace nvidia
