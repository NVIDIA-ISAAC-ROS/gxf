/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <signal.h>

// Disable breakpad for QNX
// execinfo.h is not available in the QNX toolchain
#ifdef __linux__
#include "client/linux/handler/exception_handler.h"
#endif

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/backtrace.hpp"
#include "common/logger.hpp"
#include "gflags/gflags.h"
#include "gxf/core/gxf.h"

DEFINE_string(app, "", "GXF app file to execute. Multiple files can be comma-separated");
DEFINE_string(app_root, "", "Root path for GXF app and subgraph files with relative path");
DEFINE_string(manifest, "",
              "GXF manifest file with extensions. Multiple files can be comma-separated");
DEFINE_int32(severity, -1,
             "Set log severity levels: 0=None, 1=Error, 2=Warning, 3=Info,"
             " 4=Debug, 5=Verbose. Default: Info");
DEFINE_string(log_file_path, "", "Path to a file for logging.");
DEFINE_string(graph_directory, "", "Path to a directory for searching graph files.");
DEFINE_bool(run, true, "Executes GXF application. Can be set to 'false' to verify extension and"
            " application loading.");

// Command line option for parameter override
uint32_t kNumOverrides = 32;
constexpr char override_option[] = "--param";

// Global context for signal() to interrupt with Ctrl+C
gxf_context_t s_signal_context;

// Split strings
std::vector<std::string> SplitStrings(const std::string& list_of_files) {
  std::vector<std::string> filenames;

  for (size_t begin = 0;;) {
    const size_t end = list_of_files.find(',', begin);
    if (end == std::string::npos) {
      if (begin == 0 && !list_of_files.empty()) {
        filenames.push_back(list_of_files);
      } else if (!list_of_files.substr(begin).empty()) {
        filenames.push_back(list_of_files.substr(begin));
      }
      break;
    } else {
      filenames.push_back(list_of_files.substr(begin, end - begin));
      begin = end + 1;
    }
  }

  return filenames;
}

// Loads extension manifest(s)
gxf_result_t LoadExtensionManifest(gxf_context_t context, const std::string& list_of_files) {
  const std::vector<std::string> manifest_filenames = SplitStrings(list_of_files);

  if (manifest_filenames.empty()) {
    GXF_LOG_ERROR("At least one manifest file has to be specified using -manifest");
    return GXF_FILE_NOT_FOUND;
  }

  std::vector<const char*> manifest_filenames_cstr(manifest_filenames.size());
  for (size_t i = 0; i < manifest_filenames.size(); i++) {
    manifest_filenames_cstr[i] = manifest_filenames[i].c_str();
  }

  const GxfLoadExtensionsInfo load_extension_info{
      nullptr, 0, manifest_filenames_cstr.data(),
      static_cast<uint32_t>(manifest_filenames_cstr.size()), nullptr};
  return GxfLoadExtensions(context, &load_extension_info);
}

// Loads application graph file(s)
gxf_result_t LoadApplication(gxf_context_t context,
                             const std::string& list_of_files,
                             const std::string& graph_directory,
                             const char* params_override[] = nullptr,
                             uint32_t num_override = 0) {
  const auto filenames = SplitStrings(list_of_files);

  if (filenames.empty()) {
    GXF_LOG_ERROR("At least one application file has to be specified using -app");
    return GXF_FILE_NOT_FOUND;
  }

  GxfGraphSetRootPath(context, FLAGS_app_root.c_str());
  for (const auto& filename : filenames) {
    std::string filepath = filename;
    if (!graph_directory.empty()) {
      filepath = graph_directory + "/" + filename;
    }
    GXF_LOG_INFO("Loading app: '%s'", filepath.c_str());
    const gxf_result_t code = GxfGraphLoadFile(context, filepath.c_str(),
                                params_override, num_override);
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


void ParseParamOverride(int* argc, char** argv, std::vector<std::string>& params) {
  int numParamLeft = 1;
  int numEntries = *argc;

  for (int idx = 1; idx < numEntries; idx++) {
    std::string arg = argv[idx];

    // check if the command option is "--param"
    if (arg.compare(0, strlen(override_option), override_option) == 0) {
      params.push_back(arg.substr(strlen(override_option) + 1,
                       arg.length() - strlen(override_option) - 1));

      // Adjust number of arguments
      *argc -= 1;
    } else {
      numParamLeft++;
    }

    if (numParamLeft != idx + 1) {
      // Remove the options from the list
      argv[numParamLeft] = argv[idx + 1];
    }
  }
}

int main(int argc, char** argv) {
  // execinfo.h is not available in the QNX toolchain
  #ifdef __linux__
  google_breakpad::MinidumpDescriptor descriptor("/tmp");
  google_breakpad::ExceptionHandler eh(descriptor, NULL, OnMinidump, NULL, true, -1);
  #endif

  std::vector<std::string> params_override;
  ParseParamOverride(&argc, argv, params_override);
  const char* params[kNumOverrides];
  for (size_t idx = 0; idx < params_override.size(); idx++) {
    GXF_LOG_INFO("param [%ld]: (%s)", idx, params_override[idx].c_str());
    params[idx] = params_override[idx].c_str();
  }

  gxf_result_t code;

  GXF_LOG_INFO("Creating context");
  gxf_context_t context;
  code = GxfContextCreate(&context);
  s_signal_context = context;
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("GxfContextCreate Error: %s", GxfResultStr(code));
    return 1;
  }

  gxf_runtime_info info;
  info.num_extensions = 0;
  if (GxfRuntimeInfo(context, &info) == GXF_SUCCESS) {
    gflags::SetVersionString(info.version);
  }

  const std::string usage{
        "Usage: gxe -app <graph_file_1>,...,<graph_file_n> -manifest <manifest_file>\n"
        "GXE is a runtime used to run GXF spec based graph applications using extensions\n"
        "which are loaded via a manifest file. Run 'gxe --help' for more options\n"
  };
  gflags::SetUsageMessage(usage);

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_severity < -1 || FLAGS_severity > 5) {
    GXF_LOG_ERROR("Invalid log severity level: %d", FLAGS_severity);
    return 1;
  }
  // Retrieve the current severity level if it has not been set yet.
  // The default logging severity is INFO, but it can be modified using the
  // GXF_LOG_LEVEL environment variable.
  if (FLAGS_severity == -1) {
    // Obtain the current severity level and assign it to FLAGS_severity
    GxfGetSeverity(context, reinterpret_cast<gxf_severity_t*>(&FLAGS_severity));
  }

  code = GxfSetSeverity(context, static_cast<gxf_severity_t>(FLAGS_severity));
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("GxfSetSeverity Error: %s", GxfResultStr(code));
    return 1;
  }

  FILE* fp = NULL;
  if (!FLAGS_log_file_path.empty()) {
    const char* log_file_path = FLAGS_log_file_path.c_str();
    fp = fopen(log_file_path, "w+");
    if (fp != NULL) {
      code = GxfRedirectLog(context, fp);
      if (code != GXF_SUCCESS) {
        GXF_LOG_ERROR("GxfRedirectLog Error: %s", GxfResultStr(code));
        GXF_LOG_INFO("Destroying context");
        code = GxfContextDestroy(context);
        if (code != GXF_SUCCESS) {
          GXF_LOG_ERROR("GxfContextDestroy Error: %s", GxfResultStr(code));
          return 1;
        }
        return 1;
      }
    } else {
      GXF_LOG_ERROR("Couldn't open/create the log file: %s\n", log_file_path);
      GXF_LOG_INFO("Destroying context");
      code = GxfContextDestroy(context);
      if (code != GXF_SUCCESS) {
        GXF_LOG_ERROR("GxfContextDestroy Error: %s", GxfResultStr(code));
        return 1;
      }
      return 1;
    }
  }

  code = LoadExtensionManifest(context, FLAGS_manifest);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("LoadExtensionManifest Error: %s", GxfResultStr(code));
    return 1;
  }

  if (params_override.size() == 0) {
    code = LoadApplication(context, FLAGS_app, FLAGS_graph_directory);
  } else {
    code = LoadApplication(context, FLAGS_app, FLAGS_graph_directory,
                          &params[0], params_override.size());
  }

  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("LoadApplication Error: %s", GxfResultStr(code));
    return 1;
  }

  if (FLAGS_run) {
    GXF_LOG_INFO("Initializing...");
    code = GxfGraphActivate(context);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfGraphActivate Error: %s", GxfResultStr(code));
      return 1;
    }

    GXF_LOG_INFO("Running...");
    code = GxfGraphRunAsync(context);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfGraphRunAsync Error: %s", GxfResultStr(code));
      return 1;
    }

    signal(SIGINT, [](int signum) {
      gxf_result_t code = GxfGraphInterrupt(s_signal_context);
      if (code != GXF_SUCCESS) {
        GXF_LOG_ERROR("GxfGraphInterrupt Error: %s", GxfResultStr(code));
        GXF_LOG_ERROR("Send interrupt once more to terminate immediately");
        signal(SIGINT, SIG_DFL);
      }
    });

    code = GxfGraphWait(context);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfGraphWait Error: %s", GxfResultStr(code));
      return 1;
    }

    GXF_LOG_INFO("Deinitializing...");
    code = GxfGraphDeactivate(context);
    if (code != GXF_SUCCESS) {
      GXF_LOG_ERROR("GxfGraphDeactivate Error: %s", GxfResultStr(code));
      return 1;
    }
  }

  GXF_LOG_INFO("Destroying context");
  code = GxfContextDestroy(context);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("GxfContextDestroy Error: %s", GxfResultStr(code));
    return 1;
  }
  GXF_LOG_INFO("Context destroyed.");

  if (fp != NULL) {
    GXF_LOG_INFO("Closing log file...");
    int log_code = fclose(fp);
    if (log_code != 0) {
      std::fprintf(stderr, "\033[1;31m");
      std::fprintf(stderr,
                   "==============================================================================="
                   "=====================\n");  // NOLINT
      std::fprintf(stderr, "Error closing log file: %s\n", FLAGS_log_file_path.c_str());
      std::fprintf(stderr, "\033[0m");
      return 1;
    }
  }
  return 0;
}
