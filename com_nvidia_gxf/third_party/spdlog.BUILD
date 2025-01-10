"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "spdlog",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/spdlog/**/*.h"]),
    copts = ["-DSPDLOG_COMPILED_LIB"],
    defines = [],
    include_prefix = "spdlog",
    includes = [],
    linkopts = [],
    strip_include_prefix = "include/spdlog",
    visibility = ["//visibility:public"],
    alwayslink = True,
)

