/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cxxabi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Disable backtrace for QNX since execinfo.h is not available in the QNX toolchain
#ifdef __linux__
#include <execinfo.h>
#endif

#include "common/backtrace.hpp"

// Extracts the demangled function name from a backtrace line
// If successful `demangled` will be reallocated to contain the necessary size.
// Returns the a pointer with the demangled function name; or nullptr if not successful.

char* DemangleBacktraceLine(const char* text, char** demangled, size_t* demangled_size) {
  // Strings from backtrace have the form A(X+B) [P] with A,X,B,P strings
  // X is the mangled function name which we want to demangle to create a better message
  const char* p1 = std::strchr(text, '(') + 1;
  const char* p2 = std::strchr(p1, '+');
  if (p1 == nullptr || p2 == nullptr) {
    return nullptr;
  }
  // Copy mangled name as null-terminated string to a new buffer
  const size_t mangled_size = p2 - p1;
  std::string mangled(mangled_size + 1, 0);  // +1 for null terminator
  mangled.assign(p1, mangled_size);
  // demangle the name
  int status;
  char* result = abi::__cxa_demangle(mangled.c_str(), *demangled, demangled_size, &status);
  if (status != 0) {
    result = nullptr;
  } else {
    *demangled = result;
  }

  return result;
}

// Print the stacktrace with demangled function names (if possible)
void PrettyPrintBacktrace() {
  // Disable backtrace for QNX since execinfo.h is not available in the QNX toolchain
  #ifdef __linux__
  void* array[32];
  const size_t size = backtrace(array, sizeof(array));
  char** ptr = backtrace_symbols(array, size);
  size_t demangled_size = 256;
  char* demangle_buffer = static_cast<char*>(std::malloc(demangled_size));
  for (size_t i = 1; i < size; i++) {
    char* buffer = DemangleBacktraceLine(ptr[i], &demangle_buffer, &demangled_size);
    std::fprintf(stderr, "\033[1m#%02ld\033[0m ", i);
    if (buffer == nullptr) {
      std::fprintf(stderr, "\033[2m%s\033[0m\n", ptr[i]);
    } else {
      std::fprintf(stderr, "%s \033[2m%s\033[0m\n", buffer, ptr[i]);
    }
  }
  std::free(demangle_buffer);
  #endif
}

