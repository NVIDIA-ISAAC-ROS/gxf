/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_GXF_GXF_STD_GEMS_UTILS_STORAGE_SIZE_HPP_
#define NVIDIA_GXF_GXF_STD_GEMS_UTILS_STORAGE_SIZE_HPP_

#include <iostream>
#include <string>

#include "gxf/core/expected.hpp"

namespace nvidia {
namespace gxf {

class StorageSize {
 public:
  // Conversion constants
  static constexpr size_t KB = 1024;
  static constexpr size_t MB = 1024 * KB;
  static constexpr size_t GB = 1024 * MB;
  static constexpr size_t TB = 1024 * GB;

  // Convert from other units to bytes
  static size_t toBytesFromKB(size_t kilobytes) { return kilobytes * KB; }

  static size_t toBytesFromMB(size_t megabytes) { return megabytes * MB; }

  static size_t toBytesFromGB(size_t gigabytes) { return gigabytes * GB; }

  static size_t toBytesFromTB(size_t terabytes) { return terabytes * TB; }

  // Convert from bytes to other units
  static double fromBytesToKB(size_t bytes) { return static_cast<double>(bytes) / KB; }

  static double fromBytesToMB(size_t bytes) { return static_cast<double>(bytes) / MB; }

  static double fromBytesToGB(size_t bytes) { return static_cast<double>(bytes) / GB; }

  static double fromBytesToTB(size_t bytes) { return static_cast<double>(bytes) / TB; }

  /// @brief Parses given text to return the desired storage size in bytes.
  ///
  /// @param text Text containing number and time-units, to be parsed for storage size
  /// @param cid cid of component for which text is being parsed
  /// @return size in bytes if successful, or otherwise one of the GXF error codes.
  static Expected<size_t> ParseStorageSizeString(std::string text, const gxf_uid_t& cid);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_GXF_STD_GEMS_UTILS_STORAGE_SIZE_HPP_
