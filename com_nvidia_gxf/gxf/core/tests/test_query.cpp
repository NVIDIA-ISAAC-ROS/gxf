/*
Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/core/gxf.h"

#include <complex>

#include <string.h>

#include "gtest/gtest.h"

TEST(Query, Query) {
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  // Runtime Info
  gxf_runtime_info info;
  info.num_extensions = 0;
  ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_SUCCESS);
  ASSERT_EQ(strcmp(info.version, "4.1.0"), 0);
  ASSERT_EQ(info.num_extensions, 0);

  // Extension Info
  constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
  const GxfLoadExtensionsInfo info_1{&kStdExtensionFilename, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info_1), GXF_SUCCESS);

  info.num_extensions = 0;
  ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_RESULT_ARRAY_TOO_SMALL);
  ASSERT_EQ(info.num_extensions, 1);
  std::vector<gxf_tid_t> extensions;
  extensions.resize(info.num_extensions);
  info.extensions = extensions.data();
  ASSERT_EQ(GxfRuntimeInfo(context, &info), GXF_SUCCESS);

  // Extension Info
  gxf_tid_t libext_std_tid = info.extensions[0];
  gxf_extension_info_t ext_info;

  std::vector<gxf_tid_t> component_tid_list(128);
  ext_info.num_components = 128;
  ext_info.components = component_tid_list.data();
  ASSERT_EQ(GxfExtensionInfo(context, libext_std_tid, &ext_info), GXF_SUCCESS);
  component_tid_list.resize(ext_info.num_components);

  ASSERT_GT(ext_info.num_components, 20);
  ASSERT_EQ(strcmp(ext_info.author, "NVIDIA"), 0);
  ASSERT_EQ(strcmp(ext_info.name, "StandardExtension"), 0);
  ASSERT_EQ(strcmp(ext_info.version, "2.6.0"), 0);

  // Component Info
  const gxf_tid_t comp_tid = ext_info.components[15];  // DoubleBufferTransmitter
  ASSERT_EQ(comp_tid.hash1, 0x0c3c0ec777f14389);
  ASSERT_EQ(comp_tid.hash2, 0xaef16bae85bddc13);
  gxf_component_info_t comp_info;
  std::vector<const char*> parameter_names(128);
  comp_info.num_parameters = 128;
  comp_info.parameters = parameter_names.data();
  ASSERT_EQ(GxfComponentInfo(context, comp_tid, &comp_info), GXF_SUCCESS);

  ASSERT_EQ(strcmp(comp_info.base_name, "nvidia::gxf::Transmitter"), 0);
  ASSERT_EQ(strcmp(comp_info.type_name, "nvidia::gxf::DoubleBufferTransmitter"), 0);
  ASSERT_NE(comp_info.description, nullptr);

  // Param Info
  ASSERT_EQ(comp_info.num_parameters, 2);
  ASSERT_EQ(strcmp(comp_info.parameters[0], "capacity"), 0);
  ASSERT_EQ(strcmp(comp_info.parameters[1], "policy"), 0);

  gxf_parameter_info_t param_info;
  ASSERT_EQ(GxfParameterInfo(context, comp_tid, "capacity", &param_info), GXF_SUCCESS);
  ASSERT_EQ(param_info.type, GXF_PARAMETER_TYPE_UINT64);
  ASSERT_EQ(param_info.rank, 0);
  ASSERT_EQ(*static_cast<const uint64_t*>(param_info.default_value), 1UL);

  ASSERT_EQ(GxfParameterInfo(context, comp_tid, "policy", &param_info), GXF_SUCCESS);
  ASSERT_EQ(param_info.type, GXF_PARAMETER_TYPE_UINT64);
  ASSERT_EQ(param_info.rank, 0);
  ASSERT_EQ(*static_cast<const uint64_t*>(param_info.default_value), 2UL);

  const gxf_tid_t comp_tid_h = ext_info.components[13]; // Transmitter
  ASSERT_EQ(comp_tid_h.hash1, 0xc30cc60f0db2409d);
  ASSERT_EQ(comp_tid_h.hash2, 0x92b6b2db92e02cce);
  comp_info.num_parameters = 128;
  comp_info.parameters = parameter_names.data();

  ASSERT_EQ(GxfComponentInfo(context, comp_tid_h, &comp_info), GXF_SUCCESS);
  ASSERT_EQ(strcmp(comp_info.base_name, "nvidia::gxf::Queue"), 0);
  ASSERT_EQ(strcmp(comp_info.type_name, "nvidia::gxf::Transmitter"), 0);

  const gxf_tid_t comp_tid_c = ext_info.components[17]; // Connection
  ASSERT_EQ(comp_tid_c.hash1, 0xcc71afae5ede47e9);
  ASSERT_EQ(comp_tid_c.hash2, 0xb26760a5c750a89a);

  comp_info.num_parameters = 128;
  comp_info.parameters = parameter_names.data();
  ASSERT_EQ(GxfComponentInfo(context, comp_tid_c, &comp_info), GXF_SUCCESS);

  ASSERT_EQ(strcmp(comp_info.base_name, "nvidia::gxf::Component"), 0);
  ASSERT_EQ(strcmp(comp_info.type_name, "nvidia::gxf::Connection"), 0);
  ASSERT_NE(comp_info.description, nullptr);
  ASSERT_EQ(comp_info.num_parameters, 2);
  ASSERT_EQ(strcmp(comp_info.parameters[0], "source"), 0);
  ASSERT_EQ(strcmp(comp_info.parameters[1], "target"), 0);

  // Handle Type Info
  ASSERT_EQ(GxfParameterInfo(context, comp_tid_c, "source", &param_info), GXF_SUCCESS);
  gxf_tid_t handle_tid = param_info.handle_tid; // Transmitter
  ASSERT_EQ(handle_tid.hash1, 0xc30cc60f0db2409d);
  ASSERT_EQ(handle_tid.hash2, 0x92b6b2db92e02cce);
  ASSERT_EQ(param_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(param_info.rank, 0);

  ASSERT_EQ(GxfParameterInfo(context, comp_tid_c, "target", &param_info), GXF_SUCCESS);
  handle_tid = param_info.handle_tid; // Receiver
  ASSERT_EQ(handle_tid.hash1, 0xa47d2f62245f40fc);
  ASSERT_EQ(handle_tid.hash2, 0x90b75dc78ff2437e);
  ASSERT_EQ(param_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(param_info.rank, 0);

  constexpr const char* kGxfTestExtensionFilename = "gxf/test/extensions/libgxf_test.so";
  const GxfLoadExtensionsInfo info_2{&kGxfTestExtensionFilename, 1, nullptr, 0, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &info_2), GXF_SUCCESS);

  gxf_runtime_info rt_info;
  rt_info.num_extensions = 2;
  extensions.resize(2);
  rt_info.extensions = extensions.data();
  ASSERT_EQ(GxfRuntimeInfo(context, &rt_info), GXF_SUCCESS);

  gxf_tid_t libext_test_tid = rt_info.extensions[1]; // test extension
  gxf_extension_info_t test_ext_info;

  std::vector<gxf_tid_t> test_component_tids(128);
  test_ext_info.num_components = 128;
  test_ext_info.components = test_component_tids.data();
  ASSERT_EQ(GxfExtensionInfo(context, libext_test_tid, &test_ext_info), GXF_SUCCESS);
  test_component_tids.resize(test_ext_info.num_components);

  ASSERT_GE(test_ext_info.num_components, 18);
  ASSERT_EQ(strcmp(test_ext_info.author, "NVIDIA"), 0);
  ASSERT_EQ(strcmp(test_ext_info.name, "TestHelperExtension"), 0);
  ASSERT_EQ(strcmp(test_ext_info.version, "2.6.0"), 0);

  // Component Info
  const gxf_tid_t test_comp_tid = test_ext_info.components[1];  // RegisterParameterInterfaceTest
  ASSERT_EQ(test_comp_tid.hash1, 0xe9234c1ad5f8445c);
  ASSERT_EQ(test_comp_tid.hash2, 0xae9118bcda197032);
  gxf_component_info_t test_comp_info;
  std::vector<const char*> test_parameter_names(128);
  test_comp_info.num_parameters = 128;
  test_comp_info.parameters = test_parameter_names.data();
  ASSERT_EQ(GxfComponentInfo(context, test_comp_tid, &test_comp_info), GXF_SUCCESS);

  //Parameter Info
  gxf_parameter_info_t test_param_info;
  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "mandatory_with_default",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_EQ(*static_cast<const int64_t*>(test_param_info.default_value), 3L);
  ASSERT_TRUE(test_param_info.numeric_min);
  ASSERT_EQ(*static_cast<const int64_t*>(test_param_info.numeric_min), -100L);
  ASSERT_TRUE(test_param_info.numeric_max);
  ASSERT_EQ(*static_cast<const int64_t*>(test_param_info.numeric_max), 100L);
  ASSERT_TRUE(test_param_info.numeric_step);
  ASSERT_EQ(*static_cast<const int64_t*>(test_param_info.numeric_step), 1L);
  ASSERT_NE(test_param_info.platform_information, nullptr);
  ASSERT_EQ(strcmp(test_param_info.platform_information, "linux_x86_64, linux_aarch64"), 0);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "optional_with_default",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_EQ(*static_cast<const uint64_t*>(test_param_info.default_value), 5UL);
  ASSERT_TRUE(test_param_info.numeric_min);
  ASSERT_EQ(*static_cast<const uint64_t*>(test_param_info.numeric_min), 10L);
  ASSERT_TRUE(test_param_info.numeric_max);
  ASSERT_EQ(*static_cast<const uint64_t*>(test_param_info.numeric_max), 1000L);
  ASSERT_TRUE(test_param_info.numeric_step);
  ASSERT_EQ(*static_cast<const uint64_t*>(test_param_info.numeric_step), 10L);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "bool_default",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_TRUE(*static_cast<const bool*>(test_param_info.default_value));

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "std_string_text",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_STREQ(static_cast<const char*>(test_param_info.default_value),
              "Default value of std::string text");

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "double_default",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_EQ(*static_cast<const double*>(test_param_info.default_value), 12345.6789);
  ASSERT_TRUE(test_param_info.numeric_min);
  ASSERT_EQ(*static_cast<const double*>(test_param_info.numeric_min), -10.0);
  ASSERT_TRUE(test_param_info.numeric_max);
  ASSERT_EQ(*static_cast<const double*>(test_param_info.numeric_max), 10.0);
  ASSERT_TRUE(test_param_info.numeric_step);
  ASSERT_EQ(*static_cast<const double*>(test_param_info.numeric_step), 1.0);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "complex64",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_EQ((*static_cast<const std::complex<float>*>(test_param_info.default_value)).real(), 7.5);
  ASSERT_EQ((*static_cast<const std::complex<float>*>(test_param_info.default_value)).imag(), 3.0);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "complex128",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_TRUE(test_param_info.default_value);
  ASSERT_EQ((*static_cast<const std::complex<double>*>(test_param_info.default_value)).real(), 1.234);
  ASSERT_EQ((*static_cast<const std::complex<double>*>(test_param_info.default_value)).imag(), 5.678);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "custom_parameter",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_STRING);

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

// Test case to run query api on all the extensions/components/parameters
TEST(Query, QueryAll) {

  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_runtime_info rt_info;
  rt_info.num_extensions = 0;
  ASSERT_EQ(GxfRuntimeInfo(context, &rt_info), GXF_RESULT_ARRAY_TOO_SMALL);

  std::vector<gxf_tid_t> extensions;
  extensions.resize(rt_info.num_extensions);
  rt_info.extensions = extensions.data();
  ASSERT_EQ(GxfRuntimeInfo(context, &rt_info), GXF_SUCCESS);

  // Query all extensions
  for(size_t i = 0; i < extensions.size(); ++i) {
    gxf_tid_t ext_tid = rt_info.extensions[i];
    gxf_extension_info_t ext_info;

    std::vector<gxf_tid_t> component_tid_list(128);
    ext_info.num_components = 128;
    ext_info.components = component_tid_list.data();
    ASSERT_EQ(GxfExtensionInfo(context, ext_tid, &ext_info), GXF_SUCCESS);
    component_tid_list.resize(ext_info.num_components);

    // Query all components
    for (size_t j = 0; j < component_tid_list.size(); ++j) {
      gxf_component_info_t comp_info;
      std::vector<const char*> parameter_names(128);
      comp_info.num_parameters = 128;
      comp_info.parameters = parameter_names.data();
      gxf_tid_t component_tid = ext_info.components[j];
      ASSERT_EQ(GxfComponentInfo(context, component_tid, &comp_info), GXF_SUCCESS);
      parameter_names.resize(comp_info.num_parameters);

      // Query all parameters
      for (size_t k = 0; k < parameter_names.size(); ++k) {
        gxf_parameter_info_t param_info;
        ASSERT_EQ(GxfParameterInfo(context, component_tid, parameter_names[k],
                             &param_info), GXF_SUCCESS);
      }
    }
  }

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}

TEST(Query, Shapes) {
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
  constexpr const char* kManifestFilename = "gxf/gxe/manifest.yaml";

  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &kManifestFilename, 1, nullptr};
  ASSERT_EQ(GxfLoadExtensions(context, &load_ext_info), GXF_SUCCESS);

  gxf_tid_t libext_test_tid{0x1b99ffebc2504ced, 0x811762ac05969a50}; // test extension
  gxf_extension_info_t test_ext_info;

  std::vector<gxf_tid_t> test_component_tids(128);
  test_ext_info.num_components = 128;
  test_ext_info.components = test_component_tids.data();
  ASSERT_EQ(GxfExtensionInfo(context, libext_test_tid, &test_ext_info), GXF_SUCCESS);
  test_component_tids.resize(test_ext_info.num_components);

  const gxf_tid_t test_comp_tid = test_ext_info.components[3];  // StdParameterTest
  ASSERT_EQ(test_comp_tid.hash1, 0x405d8e062d3f45f1);
  ASSERT_EQ(test_comp_tid.hash2, 0x84c492ac9ef3c67c);
  gxf_component_info_t test_comp_info;
  std::vector<const char*> test_parameter_names(128);
  test_comp_info.num_parameters = 128;
  test_comp_info.parameters = test_parameter_names.data();
  ASSERT_EQ(GxfComponentInfo(context, test_comp_tid, &test_comp_info), GXF_SUCCESS);
  test_parameter_names.resize(test_comp_info.num_parameters);

  gxf_parameter_info_t test_param_info;
  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "integers",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 1);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_INT32);
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[0]), -1); // std::vector<int32_t>

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "rank_2_vector",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 2);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_UINT64);
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[0]), -1); // std::vector<T>
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[1]), -1); // std::vector<std::vector<T>>

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "rank_3_vector",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 3);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_UINT64);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "complex64",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 0);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_COMPLEX64);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "complex128",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 0);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_COMPLEX128);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "vector_of_handles",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 1);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[0]), -1); // std::vector<Handle<T>>
  ASSERT_EQ(test_param_info.handle_tid.hash1, 0x3cdd82d023264867); // nvidia::gxf::Allocator
  ASSERT_EQ(test_param_info.handle_tid.hash2, 0x8de2d565dbe28e03);

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "rank_1_array",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 1);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_INT32);
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[0]), 2); // std::array<T,N>

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "rank_2_array",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.rank, 2);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_UINT64);
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[0]), 2); // std::array<T,N>
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[1]), 1); // std::array<std::array<T,N1> N2>

  ASSERT_EQ(GxfParameterInfo(context, test_comp_tid, "array_of_two_handles",
                             &test_param_info), GXF_SUCCESS);
  ASSERT_EQ(test_param_info.type, GXF_PARAMETER_TYPE_HANDLE);
  ASSERT_EQ(test_param_info.rank, 1);
  ASSERT_EQ(static_cast<int32_t>(test_param_info.shape[0]), 2); // std::array<T,N>
  ASSERT_EQ(test_param_info.handle_tid.hash1, 0x3cdd82d023264867); // nvidia::gxf::Allocator
  ASSERT_EQ(test_param_info.handle_tid.hash2, 0x8de2d565dbe28e03);

  ASSERT_EQ(GxfContextDestroy(context), GXF_SUCCESS);
}
