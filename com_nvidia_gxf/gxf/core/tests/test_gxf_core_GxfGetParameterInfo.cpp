/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string.h>
#include "common/assert.hpp"
#include "gxf/core/gxf.h"

#include "gtest/gtest.h"

namespace {
constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
}  // namespace

class GxfGetParameterInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info_1));
    info.num_extensions = 1;
    extensions.resize(info.num_extensions);
    info.extensions = extensions.data();
    GXF_ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_SUCCESS);
    libext_std_tid = info.extensions[0];
    std::vector<gxf_tid_t> component_tid_list(128);
    ext_info.num_components = 128;
    ext_info.components = component_tid_list.data();
    GXF_ASSERT_SUCCESS(GxfExtensionInfo(context, libext_std_tid, &ext_info));
    GXF_ASSERT_EQ(strcmp(ext_info.name, "StandardExtension"), 0);
    component_tid = ext_info.components[15];  // doublebuffer transmitter
    std::vector<const char*> param_names(128);
    comp_info.num_parameters = 128;
    comp_info.parameters = param_names.data();
    GXF_ASSERT_SUCCESS(GxfComponentInfo(context, component_tid, &comp_info));
    GXF_ASSERT_EQ(strcmp(comp_info.type_name, "nvidia::gxf::DoubleBufferTransmitter"), 0);
  }

  void TearDown() override { GXF_ASSERT_SUCCESS(GxfContextDestroy(context)); }

  gxf_context_t context = kNullContext;
  const GxfLoadExtensionsInfo info_1{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  gxf_runtime_info info;
  std::vector<gxf_tid_t> extensions;
  gxf_tid_t libext_std_tid;
  gxf_extension_info_t ext_info;
  gxf_tid_t component_tid;
  gxf_component_info_t comp_info;
};

TEST_F(GxfGetParameterInfoTest, InvalidContext) {
  gxf_parameter_info_t info;
  GXF_ASSERT_EQ(GxfGetParameterInfo(nullptr, component_tid, "capacity", &info),
                GXF_CONTEXT_INVALID);
}

TEST_F(GxfGetParameterInfoTest, InvalidInfo) {
  GXF_ASSERT_EQ(GxfGetParameterInfo(context, component_tid, "capacity", nullptr), GXF_NULL_POINTER);
}

TEST_F(GxfGetParameterInfoTest, InvalidKey) {
  gxf_parameter_info_t info;
  GXF_ASSERT_EQ(GxfGetParameterInfo(context, component_tid, "invalid", &info),
                GXF_PARAMETER_NOT_FOUND);
}

TEST_F(GxfGetParameterInfoTest, InvalidTid) {
  gxf_parameter_info_t info;
  gxf_tid_t component_tid = ext_info.components[15];
  component_tid.hash1 = 0x0c3c0ec777f14312;
  GXF_ASSERT_EQ(GxfGetParameterInfo(context, component_tid, "capacity", &info),
                GXF_ENTITY_COMPONENT_NOT_FOUND);
}

TEST_F(GxfGetParameterInfoTest, Valid) {
  gxf_parameter_info_t info;
  GXF_ASSERT_EQ(GxfGetParameterInfo(context, component_tid, "capacity", &info), GXF_SUCCESS);
  GXF_ASSERT_EQ(info.rank, 0);
  GXF_ASSERT_EQ(info.type, gxf_parameter_type_t::GXF_PARAMETER_TYPE_UINT64);
  GXF_ASSERT_EQ(strcmp(info.headline, "Capacity"), 0);
}