/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/app/sample/ping_segment_distributed.hpp"

int main(int argc, char** argv) {
  using namespace nvidia::gxf;
  auto app = create_app<SegmentDistributed>();
  /**
   * append below pre define config files to enable different segments and worker driver config
   * Example:
   * update driver_ip for all workers
   * host_1$ bazel run //gxf/app/sample:ping_segment_distributed {absolute_path}/gxf/test/apps/ping_segment_distributed.dnw1.yaml
   * host_2$ bazel run //gxf/app/sample:ping_segment_distributed {absolute_path}/gxf/test/apps/sample/ping_segment_distributed.w2.yaml
   * The first Process runs a worker that instantiates TxSegment; and a worker that drives the life cycle all segments composed in C++ API
   * The second Process runs a worker that instantiates RxSegment
  */
  app->setConfig(argc, argv);
  app->setSeverity(GXF_SEVERITY_DEBUG);
  app->loadExtensionManifest(kGxeManifestFilename);
  app->compose();
  auto result = app->run();
  return ToResultCode(result);
}
