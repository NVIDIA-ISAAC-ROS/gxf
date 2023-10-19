/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string>

#include "gxf/std/default_extension.hpp"

#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

gxf_result_t DefaultExtension::setInfo_abi(gxf_tid_t tid, const char* name, const char* desc,
                                           const char* author, const char* version,
                                           const char* license) {
  std::string description_str{desc};
  if (description_str.length() > 256) {
      GXF_LOG_ERROR("Extension description '%s' exceeds 256 characters", desc);
      return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  std::string author_str{author};
  if (author_str.length() > 64) {
      GXF_LOG_ERROR("Extension author '%s' exceeds 64 characters", author);
      return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  std::string license_str{license};
  if (license_str.length() > 64) {
      GXF_LOG_ERROR("Extension license '%s' exceeds 64 characters", license);
      return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  tid_ = tid;
  extension_version_.assign(version);
  name_.assign(name);
  description_.assign(desc);
  author_.assign(author);
  license_.assign(license);

  // Add gxf core api here to export the data to core
  return GXF_SUCCESS;
}

gxf_result_t DefaultExtension::setDisplayInfo_abi(const char* display_name, const char* category,
                                const char* brief) {
  std::string display_name_str{display_name};
  if (display_name_str.length() > 30) {
    GXF_LOG_ERROR("Extension display name '%s' exceeds 30 characters", display_name);
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  std::string category_str{category};
  if (category_str.length() > 30) {
    GXF_LOG_ERROR("Extension category %s' exceeds 30 characters", category);
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  std::string brief_str{brief};
  if (brief_str.length() > 50) {
    GXF_LOG_ERROR("Extension brief '%s' exceeds 50 characters", brief);
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  display_name_.assign(display_name);
  category_.assign(category);
  brief_.assign(brief);

  return GXF_SUCCESS;
}

// Gets description of the extension and list of components it provides
gxf_result_t DefaultExtension::getInfo_abi(gxf_extension_info_t* info) {
  if (GxfTidIsNull(tid_)) {
    return GXF_FACTORY_INVALID_INFO;
  }
  if (info == nullptr) {
    return GXF_ARGUMENT_INVALID;
  }

  info->id = tid_;
  info->version = extension_version_.c_str();
  info->runtime_version = gxf_core_version_.c_str();
  info->name = name_.c_str();
  info->description = description_.c_str();
  info->author = author_.c_str();
  info->license = license_.c_str();
  info->display_name = display_name_.c_str();
  info->category = category_.c_str();
  info->brief = brief_.c_str();

  if (info->num_components >= entries_.size() && info->components != nullptr) {
    uint64_t num_entries = entries_.size();
    for (uint64_t i = 0; i < num_entries; ++i) {
      info->components[i] = entries_.at(i)->tid;
    }
  }
  info->num_components = entries_.size();
  return GXF_SUCCESS;
}

gxf_result_t DefaultExtension::checkInfo_abi() {
  if (GxfTidIsNull(tid_) || extension_version_.empty() ||
      gxf_core_version_.empty()) {
    return GXF_FACTORY_INVALID_INFO;
  }

  return GXF_SUCCESS;
}

// Gets description of specified component (No parameter information)
gxf_result_t DefaultExtension::getComponentInfo_abi(const gxf_tid_t tid,
                                                    gxf_component_info_t* info) {
  if (info == nullptr) {
    return GXF_ARGUMENT_INVALID;
  }
  auto entry = find(tid);
  if (!entry) {
    return GXF_ENTITY_COMPONENT_NOT_FOUND;
  }
  info->cid = tid;
  info->type_name = entry->name.c_str();
  info->base_name = !entry->base.empty() ? entry->base.c_str() : nullptr;
  info->is_abstract = entry->allocator == nullptr;
  info->description = entry->description.c_str();
  info->display_name = entry->display_name.c_str();
  info->brief = entry->brief.c_str();

  return GXF_SUCCESS;
}

gxf_result_t DefaultExtension::registerComponents_abi(gxf_context_t context) {
  for (size_t i = 0; i < entries_.size(); i++) {
    const auto& entry = entries_.at(i).value();
    const gxf_result_t result =
        GxfRegisterComponent(context, entry.tid, entry.name.c_str(), entry.base.c_str());
    if (result != GXF_SUCCESS) {
      return result;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t DefaultExtension::getComponentTypes_abi(gxf_tid_t* pointer, size_t* size) {
  if (pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (size == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (*size < entries_.size()) {
    *size = entries_.size();  // return the actual size required
    return GXF_RESULT_ARRAY_TOO_SMALL;
  }

  for (size_t i = 0; i < entries_.size(); i++) {
    pointer[i] = entries_.at(i)->tid;
  }
  *size = entries_.size();

  return GXF_SUCCESS;
}

gxf_result_t DefaultExtension::allocate_abi(gxf_tid_t tid, void** out_pointer) {
  if (out_pointer == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  auto entry = find(tid);
  if (!entry) {
    return GXF_FACTORY_UNKNOWN_TID;
  }

  ComponentAllocator* allocator = entry->allocator.get();
  if (allocator == nullptr) {
    return GXF_FACTORY_ABSTRACT_CLASS;
  }

  Expected<void*> result = allocator->allocate();
  if (result) {
    *out_pointer = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t DefaultExtension::deallocate_abi(gxf_tid_t tid, void* pointer) {
  auto entry = find(tid);
  if (!entry) {
    return GXF_FACTORY_UNKNOWN_TID;
  }
  Expected<void> result = entry->allocator->deallocate(pointer);
  return result ? GXF_SUCCESS : result.error();
}

gxf_result_t DefaultExtension::getParameterInfo_abi(gxf_context_t context, const gxf_tid_t cid,
                               const char* key, gxf_parameter_info_t* info) {
  if (info == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  return GxfGetParameterInfo(context, cid, key, info);
}

Expected<DefaultExtension::Entry&> DefaultExtension::find(const gxf_tid_t& tid) {
  for (size_t i = 0; i < entries_.size(); i++) {
    Entry& entry = entries_.at(i).value();
    if (entry.tid == tid) {
      return entry;
    }
  }
  return Unexpected{GXF_QUERY_NOT_FOUND};
}

}  // namespace gxf
}  // namespace nvidia
