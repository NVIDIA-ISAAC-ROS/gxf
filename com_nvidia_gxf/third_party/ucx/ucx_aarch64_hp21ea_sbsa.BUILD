"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "ucx_aarch64_hp21ea_sbsa",
    srcs = [
        "ucx-install-with-cuda/lib/libucm.so",
        "ucx-install-with-cuda/lib/libucm.so.0",
        "ucx-install-with-cuda/lib/libucm.so.0.0.0",
        "ucx-install-with-cuda/lib/libucp.so",
        "ucx-install-with-cuda/lib/libucp.so.0",
        "ucx-install-with-cuda/lib/libucp.so.0.0.0",
        "ucx-install-with-cuda/lib/libucs_signal.so",
        "ucx-install-with-cuda/lib/libucs_signal.so.0",
        "ucx-install-with-cuda/lib/libucs_signal.so.0.0.0",
        "ucx-install-with-cuda/lib/libucs.so",
        "ucx-install-with-cuda/lib/libucs.so.0",
        "ucx-install-with-cuda/lib/libucs.so.0.0.0",
        "ucx-install-with-cuda/lib/libuct.so",
        "ucx-install-with-cuda/lib/libuct.so.0",
        "ucx-install-with-cuda/lib/libuct.so.0.0.0",
    ],
    hdrs = [
        "ucx-install-with-cuda/include/ucm/api/ucm.h",
        "ucx-install-with-cuda/include/ucp/api/ucp_compat.h",
        "ucx-install-with-cuda/include/ucp/api/ucp_def.h",
        "ucx-install-with-cuda/include/ucp/api/ucp.h",
        "ucx-install-with-cuda/include/ucp/api/ucp_version.h",
        "ucx-install-with-cuda/include/ucs/memory/memory_type.h",
        "ucx-install-with-cuda/include/ucs/sys/compiler_def.h",
        "ucx-install-with-cuda/include/ucs/type/status.h",
        "ucx-install-with-cuda/include/ucs/config/types.h",
        "ucx-install-with-cuda/include/ucs/type/thread_mode.h",
        "ucx-install-with-cuda/include/ucs/type/cpu_set.h",
    ],
    includes = ["include"],
    linkopts = [
        "-Wl,--no-as-needed," +
        "--as-needed",
    ],
    deps = ["@libnuma1_aarch64"],
    strip_include_prefix = "ucx-install-with-cuda/include",
    visibility = ["//visibility:public"],
)