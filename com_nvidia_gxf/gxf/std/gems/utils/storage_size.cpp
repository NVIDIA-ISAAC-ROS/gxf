/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/gems/utils/storage_size.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace nvidia {
namespace gxf {

Expected<size_t> StorageSize::ParseStorageSizeString(std::string text, const gxf_uid_t& cid) {
  // Parse the value as the first part of the string
  char* suffix_pointer;
  const size_t value = std::strtod(text.c_str(), &suffix_pointer);
  if (!std::isfinite(value) || suffix_pointer == text.c_str()) {
    GXF_LOG_ERROR("[C%05zu] given value '%s' is not a number", cid, text.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // Value must be positive
  if (value <= 0) {
    GXF_LOG_ERROR("[C%05zu] storage size '%s' must be positive", cid, text.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // Check for unit type.
  std::string suffix_string = text.substr(suffix_pointer - text.c_str());
  if (!suffix_string.empty() && suffix_string.front() == ' ') {
    // Discard a single leading space.
    suffix_string.erase(0, 1);
  }

  std::string suffix_upper = suffix_string;
  std::transform(suffix_upper.begin(), suffix_upper.end(), suffix_upper.begin(), ::toupper);

  if (suffix_string == "MB") {
    // If unit is MB need to convert final result to bytes by inverting the value.
    return StorageSize::toBytesFromMB(value);
  } else if (suffix_string == "KB") {
    // If unit is KB
    return StorageSize::toBytesFromKB(value);
  } else if (suffix_string == "B" || suffix_string == "") {
    // Unit is bytes
    return StorageSize::toBytesFromMB(value);
  } else if (suffix_string == "GB") {
    // Unit is GB
    return StorageSize::toBytesFromGB(value);
  } else if (suffix_string == "TB") {
    // Unit is TB
    return StorageSize::toBytesFromTB(value);
  } else {
    GXF_LOG_ERROR("[C%05zu] Invalid storage size '%s'. Unexpected suffix '%s'.", cid,
                  text.c_str(), suffix_string.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}

}  // namespace gxf
}  // namespace nvidia
