"""
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
)

load("//gxf:pybind.bzl", "pybind_library")

exports_files([
    "__init__.py",
])

pybind_library(
    name = "cuda_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/cuda",
    ],
    cc_srcs = [
        "bindings/cuda.cpp",
    ],
)

py_library(
    name = "cuda_extension",
    srcs = [
        "__init__.py",
        "Components.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
      "//gxf/cuda:cuda_pybind",
    ],
)

nv_gxf_cc_library(
    name = "cuda_lib",
    visibility = ["//visibility:public"],
    deps = [
        "cuda_buffer",
        "cuda_common",
        "cuda_event",
        "cuda_stream",
        "cuda_stream_id",
        "cuda_stream_pool",
        "cuda_stream_sync",
        "cuda_scheduling_terms",
        "stream_ordered_allocator",
        "//gxf/core:gxf",
    ],
)

nv_gxf_cc_extension(
    name = "cuda",
    srcs = ["cuda.cpp"],
    deps = [
        "cuda_lib",
        "//gxf/std:extension_factory_helper",
    ],
    ext_deps = ["//gxf/std"],
)

nv_gxf_cc_library(
    name = "stream_ordered_allocator",
    srcs = ["stream_ordered_allocator.cpp"],
    hdrs = ["stream_ordered_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_allocator",
        "cuda_common",
        "cuda_event",
        "cuda_stream",
        "cuda_stream_id",
        "//gxf/core:gxf",
    ],
)

nv_gxf_cc_library(
    name = "cuda_buffer",
    hdrs = ["cuda_buffer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_allocator",
        "cuda_common",
        "cuda_event",
        "cuda_stream_pool",
        "cuda_stream_id",
        "//gxf/core:gxf",
        "//gxf/std:memory_buffer",
    ],
)

nv_gxf_cc_library(
    name = "cuda_stream",
    srcs = ["cuda_stream.cpp"],
    hdrs = ["cuda_stream.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_allocator",
        "cuda_common",
        "cuda_event",
        "cuda_stream_id",
        "//gxf/core:gxf",
    ],
)

nv_gxf_cc_library(
    name = "cuda_stream_pool",
    srcs = ["cuda_stream_pool.cpp"],
    hdrs = ["cuda_stream_pool.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_common",
        "cuda_stream",
        "//gxf/core:gxf_cpp_api",
        "//gxf/core:parameter",
        "//gxf/core:registrar",
        "//gxf/std:allocator",
        "//gxf/std:resources",
        "//third_party:cudart",
    ],
)

nv_gxf_cc_library(
    name = "cuda_stream_id",
    hdrs = ["cuda_stream_id.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_common",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "cuda_common",
    hdrs = ["cuda_common.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
        "//third_party:cudart",
    ],
)

nv_gxf_cc_library(
    name = "cuda_event",
    srcs = ["cuda_event.cpp"],
    hdrs = ["cuda_event.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_common",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "cuda_stream_sync",
    srcs = ["cuda_stream_sync.cpp"],
    hdrs = ["cuda_stream_sync.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_common",
        "cuda_stream",
        "cuda_stream_id",
        "//gxf/core:gxf_cpp_api",
        "//gxf/core:parameter",
        "//gxf/core:registrar",
        "//gxf/std:codelet",
    ],
)

nv_gxf_cc_library(
    name = "cuda_allocator",
    hdrs = ["cuda_allocator.hpp"],
    srcs = ["cuda_allocator.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/std:allocator",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/utils:storage_size",
        "//third_party:cudart",
    ],
)

nv_gxf_cc_library(
    name = "cuda_scheduling_terms",
    srcs = ["cuda_scheduling_terms.cpp"],
    hdrs = ["cuda_scheduling_terms.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cuda_buffer",
        "cuda_stream_id",
        "cuda_stream_pool",
        "//gxf/core:gxf",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_cuda_ext",
    badges = [""],
    extension = "cuda",
    uuid = "d63a98fa-7882-11eb-a917-b38f664f399c",
    version = "2.6.0",
    headers = [
        "BUILD.public",
        "cuda_allocator.hpp",
        "cuda_buffer.hpp",
        "cuda_common.hpp",
        "cuda_event.hpp",
        "cuda_stream.hpp",
        "cuda_stream_id.hpp",
        "cuda_stream_pool.hpp",
    ],
    python_bindings = [
        "//gxf/cuda:cuda_pybind.so",
    ],
    python_sources = [
        "//gxf/cuda:__init__.py",
        "//gxf/cuda:Components.py",
        ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    url = "www.nvidia.com",
    compute_dependencies = ["cuda"],
    local_dependencies = ["//gxf/std:register_std_ext"],
    priority = "1",
    visibility = ["//visibility:public"],
)
