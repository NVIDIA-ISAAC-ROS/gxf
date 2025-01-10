# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

if(BUILD_TESTING)
    gxf_add_gxe_tests(
        GROUP_NAME PingStatistics
        APP_FILES
            test_ping_statistics.yaml
        EXT_DEPENDS
            GXF::std
            GXF::sample
            GXF::test_extension
            GXF::grpc_ext
            GXF::http
    )

    # TODO: Test appears to have no exit condition
    # add_test(
    #     NAME GxfTestRuntimeConfig
    #     COMMAND $<TARGET_FILE:gxe>
    #         -app ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime_config.yaml
    #         -manifest ${CMAKE_CURRENT_BINARY_DIR}/manifest.yaml
    # )
endif()
