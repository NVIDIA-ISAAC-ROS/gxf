/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_GXF_STD_GEMS_UTILS_TIME_HPP_
#define NVIDIA_GXF_GXF_STD_GEMS_UTILS_TIME_HPP_

#include <chrono>
#include <string>
#include "gxf/core/expected.hpp"

namespace nvidia {
namespace gxf {

/// @brief Converts time in seconds to a timestamp in nanoseconds
int64_t TimeToTimestamp(double time);

/// @brief Converts a timestamp Nanoseconds to seconds
double TimestampToTime(int64_t timestamp);

/// @brief Parses given text to return the desired period in nanoseconds.
///
/// @param text Text containing number and time-units, to be parsed for desired period
/// @param cid cid of component for which text is being parsed
/// @return Period in nanoseconds if successful, or otherwise one of the GXF error codes.
Expected<int64_t> ParseRecessPeriodString(std::string text, const gxf_uid_t& cid);

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_GXF_STD_GEMS_UTILS_TIME_HPP_
