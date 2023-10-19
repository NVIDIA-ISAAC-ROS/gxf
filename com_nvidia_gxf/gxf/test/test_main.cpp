/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Disable breakpad for QNX
// execinfo.h is not available in the QNX toolchain
#ifdef __linux__
#include <execinfo.h>
#include "client/linux/handler/exception_handler.h"
#endif

#include <cstring>

#include <sstream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

#include "gxf/core/gxf.h"
#include "common/assert.hpp"
#include "common/backtrace.hpp"
#include "common/logger.hpp"

DEFINE_string(app, "", "GXF app file to execute");
DEFINE_string(manifest, "", "GXF extension manifest");

// Split strings
std::vector<std::string> SplitStrings(const std::string& list_of_files) {
  std::vector<std::string> filenames;
  char delimiter = ',';
  std::istringstream stream(list_of_files);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
      filenames.push_back(item);
  }

  return filenames;
}

// Loads application graph file(s)
gxf_result_t LoadApplication(gxf_context_t context, const std::string& list_of_files) {
  const auto filenames = SplitStrings(list_of_files);

  if (filenames.empty()) {
    GXF_LOG_ERROR("Atleast one application file has to be specified using -app");
    return GXF_FILE_NOT_FOUND;
  }

  for (const auto& filename : filenames) {
    GXF_LOG_INFO("Loading app: '%s'", filename.c_str());
    const gxf_result_t code = GxfGraphLoadFile(context, filename.c_str());
    if (code != GXF_SUCCESS) { return code; }
  }

  return GXF_SUCCESS;
}

// execinfo.h is not available in the QNX toolchain
#ifdef __linux__
// Crash handler call back function
static bool OnMinidump(const google_breakpad::MinidumpDescriptor& descriptor, void* context,
                       bool succeeded) {
  // Print header
  std::fprintf(stderr, "\033[1;31m");
  std::fprintf(stderr, "====================================================================================================\n");  // NOLINT
  std::fprintf(stderr, "|                            GXF terminated unexpectedly                                           |\n");  // NOLINT
  std::fprintf(stderr, "====================================================================================================\n");  // NOLINT
  std::fprintf(stderr, "\033[0m");

  PrettyPrintBacktrace();

  // Print footer with mention to minidump
  std::fprintf(stderr, "\033[1;31m");
  std::fprintf(stderr, "====================================================================================================\n");  // NOLINT
  std::fprintf(stderr, "Minidump written to: %s\n", descriptor.path());
  std::fprintf(stderr, "\033[0m");
  return succeeded;
}
#endif  // __linux__

int main(int argc, char **argv) {
  // execinfo.h is not available in the QNX toolchain
  #ifdef __linux__
  google_breakpad::MinidumpDescriptor descriptor("/tmp");
  google_breakpad::ExceptionHandler eh(descriptor, NULL, OnMinidump, NULL, true, -1);
  #endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);

  gxf_context_t context;
  GXF_LOG_INFO("Creating context");
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const char* manifest_filename = FLAGS_manifest.c_str();
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &manifest_filename, 1, nullptr};
  GXF_LOG_INFO("Loading extensions");
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_ext_info));
  GXF_LOG_INFO("Loading graph file %s", FLAGS_app.c_str());
  GXF_ASSERT_SUCCESS(LoadApplication(context, FLAGS_app.c_str()));
  GXF_LOG_INFO("Setting Log Level to DEBUG...");
  GXF_ASSERT_SUCCESS(GxfSetSeverity(context, gxf_severity_t::GXF_SEVERITY_DEBUG));
  GXF_LOG_INFO("Initializing...");
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_LOG_INFO("Running...");
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_LOG_INFO("Deinitializing...");
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_LOG_INFO("Destroying context");
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));

  return 0;
}
