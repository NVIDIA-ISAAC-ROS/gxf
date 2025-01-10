"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "rmm",
    srcs = [],
    hdrs = glob([
            "include/rmm/**/*.hpp",
            "include/rmm/**/*.h",
           ]),
    copts = ["LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    defines = [],
    include_prefix = "rmm",
    includes = [],
    linkopts = [],
    strip_include_prefix = "include/rmm",
    visibility = ["//visibility:public"],
    deps = [
        "@fmt",
        "@spdlog",
        "@com_nvidia_gxf//third_party:cuda_headers",
    ],
    alwayslink = True,
)
