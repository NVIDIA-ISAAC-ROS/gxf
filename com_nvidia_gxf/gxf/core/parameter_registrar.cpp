/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/core/parameter_registrar.hpp"

#include <complex>
#include <memory>
#include <string>
#include <utility>

namespace nvidia {
namespace gxf {

// Adds a parameterless type to parameter registrar. These are useful to lookup tid
// using type name when parameters of type Handle<T> are being registered
void ParameterRegistrar::addParameterlessType(const gxf_tid_t tid, std::string type_name) {
  auto ptr = std::make_unique<ComponentInfo>();
  ptr->type_name = type_name;
  component_parameters[tid] = std::move(ptr);
}

// Check if parameter registrar has a component
bool ParameterRegistrar::hasComponent(const gxf_tid_t tid) const {
  return component_parameters.find(tid) == component_parameters.end() ? false : true;
}

// Get the number of parameters in a component
size_t ParameterRegistrar::componentParameterCount(const gxf_tid_t tid) const {
  auto cit = component_parameters.find(tid);
  return cit == component_parameters.end() ? 0 : cit->second->parameter_keys.size();
}

// Get the list of parameter keys in a component
Expected<void> ParameterRegistrar::getParameterKeys(const gxf_tid_t tid, const char** keys,
                                                    size_t& count) const {
  auto cit = component_parameters.find(tid);
  if (cit == component_parameters.end()) {
    count = 0;
    return Success;
  }
  if (count < cit->second->parameter_keys.size()) {
    count = cit->second->parameter_keys.size();
    return Unexpected{GXF_RESULT_ARRAY_TOO_SMALL};
  }
  count = 0;
  for (const auto& pkey : cit->second->parameter_keys) {
    keys[count++] = pkey.c_str();
  }
  return Success;
}

// Check if a component has a parameter
Expected<bool> ParameterRegistrar::componentHasParameter(const gxf_tid_t tid,
                                                         const char* key) const {
  auto cit = component_parameters.find(tid);
  if (cit == component_parameters.end()) {
    return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
  }
  auto pit = cit->second->parameters.find(key);
  if (pit == cit->second->parameters.end()) {
    return Unexpected{GXF_PARAMETER_NOT_FOUND};
  }
  return true;
}

// Get the default value of a component
// Depending on the data type of the parameter, default value has to be
// casted into the data type specified in gxf_parameter_type_t in gxf.h
Expected<const void*> ParameterRegistrar::getDefaultValue(const gxf_tid_t tid,
                                                          const char* key) const {
  auto result = getComponentParameterInfoPtr(tid, key);
  if (!result) {
    return ForwardError(result);
  }

  auto pinfo = result.value();
  if (!pinfo->default_value.has_value()) {
    return nullptr;
  }

  const void* default_value = nullptr;
  switch (pinfo->type) {
    case GXF_PARAMETER_TYPE_STRING: {
      const std::string* ptr = reinterpret_cast<const std::string*>(pinfo->default_value.get());
      if (ptr != nullptr) {
        default_value = static_cast<const void*>(ptr->c_str());
      }
    } break;
    case GXF_PARAMETER_TYPE_FILE: {
      const FilePath* ptr = reinterpret_cast<const FilePath*>(pinfo->default_value.get());
      if (ptr != nullptr) {
        default_value = static_cast<const void*>(ptr->c_str());
      }
    } break;
    case GXF_PARAMETER_TYPE_INT8:
    case GXF_PARAMETER_TYPE_INT16:
    case GXF_PARAMETER_TYPE_INT32:
    case GXF_PARAMETER_TYPE_INT64:
    case GXF_PARAMETER_TYPE_UINT8:
    case GXF_PARAMETER_TYPE_UINT16:
    case GXF_PARAMETER_TYPE_UINT32:
    case GXF_PARAMETER_TYPE_UINT64:
    case GXF_PARAMETER_TYPE_FLOAT32:
    case GXF_PARAMETER_TYPE_FLOAT64:
    case GXF_PARAMETER_TYPE_COMPLEX64:
    case GXF_PARAMETER_TYPE_COMPLEX128:
    case GXF_PARAMETER_TYPE_BOOL: {
      default_value = pinfo->default_value.get();
    } break;
    case GXF_PARAMETER_TYPE_CUSTOM:
    case GXF_PARAMETER_TYPE_HANDLE: {
      break;
    }
    default: {
      GXF_LOG_DEBUG("no default value for parameter %s", key);
    } break;
  }

  return default_value;
}

// Load the numeric range of a parameter into gxf_parameter_info_t
// Depending on the data type of the parameter, numeric ranges value have to be
// casted into the data type specified in gxf_parameter_type_t in gxf.h
Expected<bool> ParameterRegistrar::getNumericRange(const gxf_tid_t tid, const char* key,
                                                   gxf_parameter_info_t* info) const {
  auto result_ptr = getComponentParameterInfoPtr(tid, key);
  if (!result_ptr) {
    return ForwardError(result_ptr);
  }
  auto ptr = result_ptr.value();
  if (!ptr->is_arithmetic) {
    return Unexpected(GXF_PARAMETER_NOT_NUMERIC);
  }
  switch (ptr->type) {
    case GXF_PARAMETER_TYPE_INT8: {
      return getNumericRangeImpl<int8_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_INT16: {
      return getNumericRangeImpl<int16_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_INT32: {
      return getNumericRangeImpl<int32_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_INT64: {
      return getNumericRangeImpl<int64_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_UINT8: {
      return getNumericRangeImpl<uint8_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_UINT16: {
      return getNumericRangeImpl<uint16_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_UINT32: {
      return getNumericRangeImpl<uint32_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_UINT64: {
      return getNumericRangeImpl<uint64_t>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_FLOAT32: {
      return getNumericRangeImpl<float>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_FLOAT64: {
      return getNumericRangeImpl<double>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_COMPLEX64: {
      return getNumericRangeImpl<std::complex<float>>(ptr, info);
    } break;
    case GXF_PARAMETER_TYPE_COMPLEX128: {
      return getNumericRangeImpl<std::complex<double>>(ptr, info);
    } break;
    default: {
      GXF_LOG_DEBUG("no numeric ranges for parameter %s", key);
    } break;
  }
  return false;
}

// fills the gxf_parameter_info_t struct with the info stored in the parameter registrar
// for that specific component and parameter key
Expected<void> ParameterRegistrar::getParameterInfo(const gxf_tid_t tid, const char* key,
                                                    gxf_parameter_info_t* info) const {
  auto result = getComponentParameterInfoPtr(tid, key);
  if (!result) {
    return ForwardError(result);
  }

  auto pinfo = result.value();
  info->key = pinfo->key.c_str();
  info->headline = pinfo->headline.c_str();
  info->description = pinfo->description.c_str();
  info->flags = pinfo->flags;
  info->platform_information = pinfo->platform_information.c_str();
  info->type = pinfo->type;
  info->handle_tid = pinfo->handle_tid;
  info->rank = pinfo->rank;
  // Fill shape information
  for (int i = 0; i < info->rank; ++i) {
    info->shape[i] = pinfo->shape[i];
  }

  auto default_value = getDefaultValue(tid, key);
  if (!default_value) {
    return ForwardError(default_value);
  } else {
    info->default_value = default_value.value();
  }

  if (pinfo->is_arithmetic) {
    auto num_range = getNumericRange(tid, key, info);
    if (!num_range) {
      return ForwardError(num_range);
    } else if (!num_range.value()) {
      GXF_LOG_WARNING("Failed to get numeric ranges for parameter %s", key);
    }
  }

  return Success;
}

// Returns the pointer to ComponentParameterInfo object in ComponentInfo
Expected<ParameterRegistrar::ComponentParameterInfo*>
ParameterRegistrar::getComponentParameterInfoPtr(const gxf_tid_t tid, const char* key) const {
  auto result = componentHasParameter(tid, key);
  if (!result) {
    return ForwardError(result);
  }

  auto cit = component_parameters.find(tid);
  auto pit = cit->second->parameters.find(key);
  return &pit->second;
}

Expected<void> ParameterRegistrar::registerComponentParameterImpl(
    gxf_tid_t tid, const std::string& type_name, ComponentParameterInfo& info) {
  auto it = component_parameters.find(tid);
  if (it == component_parameters.end()) {
    component_parameters[tid] = std::make_unique<ComponentInfo>();
    it = component_parameters.find(tid);
  }

  const auto jt = it->second->parameters.find(info.key);
  if (jt != it->second->parameters.end()) {
    return Unexpected{GXF_PARAMETER_ALREADY_REGISTERED};
  }

  it->second->parameter_keys.push_back(info.key);
  it->second->parameters[info.key] = info;
  it->second->type_name = type_name;
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
