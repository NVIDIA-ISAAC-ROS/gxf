# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

add_subdirectory(apps)
add_subdirectory(entity_throughput)
add_subdirectory(test_extension)

gxf_add_gtests(
    EXT_NAME Core
    SOURCES
        test_api_GxfParameterGet1DFloat64Vector.cpp
        test_api_GxfParameterGet1DFloat64VectorInfo.cpp
        test_api_GxfParameterGet1DInt32Vector.cpp
        test_api_GxfParameterGet1DInt32VectorInfo.cpp
        test_api_GxfParameterGet1DInt64Vector.cpp
        test_api_GxfParameterGet1DInt64VectorInfo.cpp
        test_api_GxfParameterGet1DUInt64Vector.cpp
        test_api_GxfParameterGet1DUInt64VectorInfo.cpp
        test_api_GxfParameterGet2DFloat64Vector.cpp
        test_api_GxfParameterGet2DFloat64VectorInfo.cpp
        test_api_GxfParameterGet2DInt32Vector.cpp
        test_api_GxfParameterGet2DInt32VectorInfo.cpp
        test_api_GxfParameterGet2DInt64Vector.cpp
        test_api_GxfParameterGet2DInt64VectorInfo.cpp
        test_api_GxfParameterGet2DUInt64Vector.cpp
        test_api_GxfParameterGet2DUInt64VectorInfo.cpp
        test_entity.cpp
        test_expected.cpp
        test_gxf_core_GxfComponentAdd.cpp
        test_gxf_core_GxfComponentAddAndGetPtr.cpp
        test_gxf_core_GxfComponentAddToInterface.cpp
        test_gxf_core_GxfComponentEntity.cpp
        test_gxf_core_GxfComponentFind.cpp
        test_gxf_core_GxfComponentFindAndGetPtr.cpp
        test_gxf_core_GxfComponentInfo.cpp
        test_gxf_core_GxfComponentName.cpp
        test_gxf_core_GxfComponentPointer.cpp
        test_gxf_core_GxfComponentType.cpp
        test_gxf_core_GxfComponentTypeId.cpp
        test_gxf_core_GxfComponentTypeName.cpp
        test_gxf_core_GxfContextCreate.cpp
        test_gxf_core_GxfContextDestroy.cpp
        test_gxf_core_GxfCreateEntity.cpp
        test_gxf_core_GxfCreateEntityAndGetItem.cpp
        test_gxf_core_GxfCreateEntityGroup.cpp
        test_gxf_core_GxfEntityActivate.cpp
        test_gxf_core_GxfEntityDestroy.cpp
        test_gxf_core_GxfEntityFindAll.cpp
        test_gxf_core_GxfEntityGetName.cpp
        test_gxf_core_GxfEntityGetRefCount.cpp
        test_gxf_core_GxfEntityFind.cpp
        test_gxf_core_GxfEntityGetStatus.cpp
        test_gxf_core_GxfEntityItemPtr.cpp
        test_gxf_core_GxfEntityRefCountDec.cpp
        test_gxf_core_GxfEntityRefCountInc.cpp
        test_gxf_core_GxfExtensionInfo.cpp
        test_gxf_core_GxfGetParameterInfo.cpp
        test_gxf_core_GxfParameterFlagTypeStr.cpp
        test_gxf_core_GxfParameterGetAsYamlNode.cpp
        test_gxf_core_GxfParameterGetBool.cpp
        test_gxf_core_GxfParameterGetFloat32.cpp
        test_gxf_core_GxfParameterGetFloat64.cpp
        test_gxf_core_GxfParameterGetHandle.cpp
        test_gxf_core_GxfParameterGetInt32.cpp
        test_gxf_core_GxfParameterGetInt64.cpp
        test_gxf_core_GxfParameterGetStr.cpp
        test_gxf_core_GxfParameterGetUInt64.cpp
        test_gxf_core_GxfParameterInfo.cpp
        test_gxf_core_GxfParameterSet1DFloat64Vector.cpp
        test_gxf_core_GxfParameterSet1DInt32Vector.cpp
        test_gxf_core_GxfParameterSet1DInt64Vector.cpp
        test_gxf_core_GxfParameterSet1DUInt64Vector.cpp
        test_gxf_core_GxfParameterSet2DFloat64Vector.cpp
        test_gxf_core_GxfParameterSet2DInt32Vector.cpp
        test_gxf_core_GxfParameterSet2DInt64Vector.cpp
        test_gxf_core_GxfParameterSet2DUInt64Vector.cpp
        test_gxf_core_GxfParameterSetBool.cpp
        test_gxf_core_GxfParameterSetFloat32.cpp
        test_gxf_core_GxfParameterSetFloat64.cpp
        test_gxf_core_GxfParameterSetHandle.cpp
        test_gxf_core_GxfParameterSetInt32.cpp
        test_gxf_core_GxfParameterSetInt64.cpp
        test_gxf_core_GxfParameterSetStr.cpp
        test_gxf_core_GxfParameterSetUInt64.cpp
        test_gxf_core_GxfParameterTypeStr.cpp
        test_gxf_core_GxfRedirectLog.cpp
        test_gxf_core_GxfUpdateEntityGroup.cpp
        test_GxfLoadExtensionMetadataFiles.cpp
        test_GxfRegisterComponent.cpp
        test_query.cpp
        test_gxf_core_GxfEntityDeactivate.cpp
        test_gxf_core_GxfGraphDeactivate.cpp
        test_gxf_core_GxfGraphActivate.cpp
        test_gxf_core_GxfGraphInterrupt.cpp
        test_gxf_core_GxfGraphLoadFile.cpp
        test_gxf_core_GxfGraphParseString.cpp
        test_gxf_core_GxfGraphSetRootPath.cpp
        test_gxf_core_GxfGraphRun.cpp
        test_gxf_core_GxfGraphWait.cpp
        test_gxf_core_GxfRuntimeInfo.cpp
        test_gxf_core_GxfParameterSetFromYaml.cpp
    DATA_FILES
        metadata_component_not_list.yaml
        metadata_component_not_map.yaml
        metadata_invalid_type_id.yaml
        metadata_missing_base_typename.yaml
        metadata_missing_typeid.yaml
        metadata_missing_typename.yaml
        metadata.yaml
        test_app_GxfComponentAddToInterface.yaml
        test_ping_dynamic_activation.yaml
    DEPENDS
        GXF::core
        GXF::std
        breakpad::breakpad
        gflags::gflags_static
        GTest::gtest_main
    BUILD_DEPENDS
        GXF::sample
        core_tests_test_extension
        gxf_core_manifest
)

# NVBug 4499812
# TODO: Provide a custom manifest with a subset of extensions for testing GXF Core only.
#
# set(GXF_CORE_TEST_MANIFEST_FILEPATH "${CMAKE_CURRENT_BINARY_DIR}/manifest.yaml")
# set(GXF_CORE_TEST_EXTENSIONS "std;sample;test::extensions")
# gxf_generate_manifest_file("${GXF_CORE_TEST_MANIFEST_FILEPATH}" "${GXF_CORE_TEST_EXTENSIONS}")
