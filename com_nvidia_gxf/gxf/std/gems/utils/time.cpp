/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/gems/utils/time.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace nvidia {
namespace gxf {

namespace {

constexpr double kNanosecondsPerSecond = 1'000'000'000.0;

}  // namespace

int64_t TimeToTimestamp(double time) {
  return static_cast<int64_t>(time * kNanosecondsPerSecond);
}

double TimestampToTime(int64_t timestamp) {
  return static_cast<double>(timestamp) / kNanosecondsPerSecond;
}

Expected<int64_t> ParseRecessPeriodString(std::string text, const gxf_uid_t& cid) {
  // Convert string to standard lower form.
  std::transform(text.begin(), text.end(), text.begin(), ::tolower);

  // Parse the value as the first part of the string
  char* suffix_pointer;
  const double value = std::strtod(text.c_str(), &suffix_pointer);
  if (!std::isfinite(value) || suffix_pointer == text.c_str()) {
    GXF_LOG_ERROR("[C%05zu] Tick period '%s' is not a number", cid, text.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // Value must be positive
  if (value <= 0.0) {
    GXF_LOG_ERROR("[C%05zu] Tick period '%s' must be positive", cid, text.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // Check for unit type.
  std::string suffix_string = text.substr(suffix_pointer - text.c_str());
  if (!suffix_string.empty() && suffix_string.front() == ' ') {
    // Discard a single leading space.
    suffix_string.erase(0, 1);
  }
  if (suffix_string == "") {
    // Default unit is ns
    return static_cast<int64_t>(value);
  } else if (suffix_string == "hz") {
    // If unit is Hz need to convert final result to seconds by inverting the value.
    return TimeToTimestamp(1.0 / value);
  } else if (suffix_string == "ms") {
    // If unit is ms the final result must be divided by 1000
    return TimeToTimestamp(0.001 * value);
  } else if (suffix_string == "s") {
    // Unit is seconds
    return TimeToTimestamp(value);
  } else {
    GXF_LOG_ERROR("[C%05zu] Invalid tick period '%s'. Unexpected suffix '%s'.", cid, text.c_str(),
                  suffix_string.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}

}  // namespace gxf
}  // namespace nvidia
