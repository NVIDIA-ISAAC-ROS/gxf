"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_test(
    name = "test_logger",
    srcs = ["test_logger.cpp"],
    deps = [
        "//gxf/logger:common_logger",
        "@gtest//:main",
    ],
)

cc_test(
    name = "test_gxf_logger",
    srcs = ["test_gxf_logger.cpp"],
    deps = [
        "//gxf/logger:gxf_logger",
        "@gtest//:main",
    ],
)
