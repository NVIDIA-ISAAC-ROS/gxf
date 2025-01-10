/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_EXTENSION_LOADER_HPP_
#define NVIDIA_GXF_STD_EXTENSION_LOADER_HPP_

#include <map>
#include <set>
#include <shared_mutex>  // NOLINT

#include "common/fixed_vector.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/std/component_factory.hpp"
#include "gxf/std/extension.hpp"

namespace nvidia {
namespace gxf {

// Loads extensions and allows to create instances of their components.
class ExtensionLoader : public ComponentFactory {
 public:
  ExtensionLoader() = default;
  ~ExtensionLoader() = default;

  ExtensionLoader(const ExtensionLoader&) = delete;
  ExtensionLoader(ExtensionLoader&&) = delete;
  ExtensionLoader& operator=(const ExtensionLoader&) = delete;
  ExtensionLoader& operator=(ExtensionLoader&&) = delete;

  gxf_result_t allocate_abi(gxf_tid_t tid, void** out_pointer) override;
  gxf_result_t deallocate_abi(gxf_tid_t tid, void* pointer) override;

  // Loads a GXF extension from the given file
  Expected<void> initialize(gxf_context_t context);
  Expected<void> load(const char* filename);
  Expected<void> load(Extension* extension, void* handle = nullptr);
  Expected<void> registerRuntimeComponent(const gxf_tid_t& component_tid,
                                          const gxf_tid_t& extension_tid);
  Expected<void> getComponentTypes(gxf_tid_t* pointer, size_t* size);
  Expected<void> unloadAll();

  // Gets list of TIDs for loaded Extensions.
  // [in/out] extension_num is for capacity of parameter extensions
  // [out] extensions is memory to write TIDs to
  Expected<void> getExtensions(uint64_t* extension_count, gxf_tid_t* extensions);

  // Gets description for specified (loaded) extension and list of components It provides
  Expected<void> getExtensionInfo(gxf_tid_t eid, gxf_extension_info_t* info);

  // Gets description of Component (No list of parameter)
  Expected<void> getComponentInfo(const gxf_tid_t tid, gxf_component_info_t* info);

  Expected<void> getParameterInfo(const gxf_tid_t cid, const char* key, gxf_parameter_info_t* info);

 private:
  gxf_context_t context_ = nullptr;
  std::set<void*> handles_;
  std::map<gxf_tid_t, Extension*> component_factory_;
  FixedVector<Extension*> factories_;
  std::map<gxf_tid_t, Extension*> extension_factory_;

  mutable std::shared_timed_mutex mutex_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_EXTENSION_LOADER_HPP_
