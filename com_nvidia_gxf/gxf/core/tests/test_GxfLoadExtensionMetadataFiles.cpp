/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/gxf_ext.h"

TEST(LoadExtensionMetadataFiles, LoadValidFile){
    gxf_context_t context = kNullContext;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

    std::vector<const char*> ext_filnames(1);
    ext_filnames[0] = "gxf/core/tests/metadata.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_SUCCESS);

    gxf_tid_t tid = GxfTidNull();
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context, "nvidia::gxf::test::DummyComponent", &tid));
}

TEST(LoadExtensionMetadataFiles, InvalidContext){
    std::vector<const char*> ext_filnames(1);
    ext_filnames[0] = "gxf/core/tests/metadata.yaml";

    ASSERT_EQ(GxfLoadExtensionMetadataFiles(nullptr, ext_filnames.data(), 1), GXF_CONTEXT_INVALID);
}

TEST(LoadExtensionMetadataFiles, LoadNULLFile){
    gxf_context_t context = kNullContext;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, nullptr, 1), GXF_ARGUMENT_NULL);
}

TEST(LoadExtensionMetadataFiles, LoadWrongFile){
    gxf_context_t context = kNullContext;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

    std::vector<const char*> ext_filnames(1);

    // file does not exist
    ext_filnames[0] = "gxf/core/tests/metadata_wrong_path.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);

    // not a YAML file
    ext_filnames[0] = "gxf/core/tests/README.md";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);
}


TEST(LoadExtensionMetadataFiles, LoadBadYAMLFile){
    gxf_context_t context = kNullContext;
    ASSERT_EQ(GxfContextCreate(&context), GXF_SUCCESS);

    std::vector<const char*> ext_filnames(1);

    // component is not a sequence in YAML
    ext_filnames[0] = "gxf/core/tests/metadata_component_not_list.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);

    // missing typename attribute in YAML
    ext_filnames[0] = "gxf/core/tests/metadata_missing_typename.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);

    // missing typeid attribute in YAML
    ext_filnames[0] = "gxf/core/tests/metadata_missing_typeid.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);

    // missing base_typename attribute in YAML
    ext_filnames[0] = "gxf/core/tests/metadata_missing_base_typename.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);

    // invalid uuid in YAML
    ext_filnames[0] = "gxf/core/tests/metadata_invalid_type_id.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);

    // component metadata is not a map in YAML
    ext_filnames[0] = "gxf/core/tests/metadata_component_not_map.yaml";
    ASSERT_EQ(GxfLoadExtensionMetadataFiles(context, ext_filnames.data(), 1), GXF_FAILURE);
}
