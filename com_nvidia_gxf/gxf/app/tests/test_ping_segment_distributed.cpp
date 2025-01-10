/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/app/tests/multi_process_helper.hpp"
#include "gxf/app/sample/ping_segment_distributed.hpp"

int processEntrance(const std::string& arg) {
  GXF_LOG_INFO("Instance is running, with arg: %s, PID: %d", arg.c_str(), getpid());
  using namespace nvidia::gxf;
  auto app = create_app<SegmentDistributed>();
  app->setConfig(arg);
  app->setSeverity(GXF_SEVERITY_DEBUG);
  app->loadExtensionManifest(nvidia::gxf::kGxeManifestFilename);
  app->compose();
  auto result = app->run();
  return ToResultCode(result);
}

int main(int argc, char** argv) {
  const std::string config1 = "gxf/test/apps/ping_segment_distributed.dnw1.yaml";
  const std::string config2 = "gxf/test/apps/ping_segment_distributed.w2.yaml";
  pid_t pid1 = processRun(processEntrance, config1);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  pid_t pid2 = processRun(processEntrance, config2);

  int exitStatus1 = processWait(pid1);
  int exitStatus2 = processWait(pid2);
  if (exitStatus1 != 0) {
    return exitStatus1;
  }
  if (exitStatus2 != 0) {
    return exitStatus2;
  }
  // Return 0 if both children exited with status 0
  return 0;
}
