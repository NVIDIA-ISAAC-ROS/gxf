/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <cstdint>
#include <cstring>

namespace nvidia {

char* TypenameAsStringGnuC(const char* pretty, char* output, int32_t max_length) {
  // Search for the = sign.
  const char* begin = std::strchr(pretty, '=');
  if (begin == nullptr) { return nullptr; }

  // Skip the space
  begin++;
  if (*begin != ' ') { return nullptr; }
  begin++;

  // Search for the closing ]
  const char* end = std::strchr(begin, ']');
  if (end == nullptr) { return nullptr; }

  // Make sure we don't exceed the maximum size
  const int32_t length = end - begin;
  if (length >= max_length) { return nullptr; }

  // We do not allow the empty string as a type name.
  if (length == 0) { return nullptr; }

  // Copy the string to the target array
  std::memcpy(output, begin, length);

  // Add a null-terminator
  output[length] = 0;

  return output;
}

}  // namespace nvidia
