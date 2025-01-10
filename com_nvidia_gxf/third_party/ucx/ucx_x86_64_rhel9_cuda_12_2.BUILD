"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "ucx_x86_64_rhel9_cuda_12_2",
    srcs = [
        "lib64/libucm.so",
        "lib64/libucm.so.0",
        "lib64/libucm.so.0.0.0",
        "lib64/libucp.so",
        "lib64/libucp.so.0",
        "lib64/libucp.so.0.0.0",
        "lib64/libucs_signal.so",
        "lib64/libucs_signal.so.0",
        "lib64/libucs_signal.so.0.0.0",
        "lib64/libucs.so",
        "lib64/libucs.so.0",
        "lib64/libucs.so.0.0.0",
        "lib64/libuct.so",
        "lib64/libuct.so.0",
        "lib64/libuct.so.0.0.0",
        "lib64/ucx/libucm_cuda.so",
        "lib64/ucx/libucm_cuda.so.0",
        "lib64/ucx/libucm_cuda.so.0.0.0",
        "lib64/ucx/libuct_cma.so",
        "lib64/ucx/libuct_cma.so.0",
        "lib64/ucx/libuct_cma.so.0.0.0",
        "lib64/ucx/libuct_cuda.so",
        "lib64/ucx/libuct_cuda.so.0",
        "lib64/ucx/libuct_cuda.so.0.0.0",
        "lib64/ucx/libuct_ib.so",
        "lib64/ucx/libuct_ib.so.0",
        "lib64/ucx/libuct_ib.so.0.0.0",
        "lib64/ucx/libuct_rdmacm.so",
        "lib64/ucx/libuct_rdmacm.so.0",
        "lib64/ucx/libuct_rdmacm.so.0.0.0",
        # "lib64/ucx/libucx_perftest_cuda.so",
        # "lib64/ucx/libucx_perftest_cuda.so.0",
        # "lib64/ucx/libucx_perftest_cuda.so.0.0.0",
    ],
    hdrs = [
        "include/ucm/api/ucm.h",
        "include/ucp/api/ucp_compat.h",
        "include/ucp/api/ucp_def.h",
        "include/ucp/api/ucp.h",
        "include/ucp/api/ucp_version.h",
        "include/ucs/memory/memory_type.h",
        "include/ucs/sys/compiler_def.h",
        "include/ucs/type/status.h",
        "include/ucs/config/types.h",
        "include/ucs/type/thread_mode.h",
        "include/ucs/type/cpu_set.h",
    ],
    includes = ["include"],
    linkopts = [
        "-Wl,--no-as-needed," +
        "--as-needed",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)