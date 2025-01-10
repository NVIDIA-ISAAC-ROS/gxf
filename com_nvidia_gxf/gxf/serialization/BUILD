"""
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

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

py_library(
    name = "serialization_extension",
    srcs = [
        "//gxf/serialization/bindings:__init__.py",
        "//gxf/serialization/bindings:cask.py",
        "//gxf/serialization:__init__.py",
        "//gxf/serialization:Components.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:core_pybind",
        "//gxf/std:tensor_pybind",
        "//gxf/std:vault_pybind",
    ],
)

nv_gxf_cc_extension(
    name = "serialization",
    srcs = ["serialization.cpp"],
    deps = [
        "component_serializer",
        "endpoint",
        "entity_recorder",
        "entity_replayer",
        "entity_serializer",
        "file",
        "serialization_buffer",
        "std_component_serializer",
        "std_entity_id_serializer",
        "std_entity_serializer",
    ],
    ext_deps = [
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "entity_replayer",
    srcs = ["entity_replayer.cpp"],
    hdrs = ["entity_replayer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "entity_serializer",
        "file_stream",
        "//gxf/std:codelet",
        "//gxf/std:scheduling_terms",
    ],
)

nv_gxf_cc_library(
    name = "entity_recorder",
    srcs = ["entity_recorder.cpp"],
    hdrs = ["entity_recorder.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "entity_serializer",
        "file_stream",
        "//gxf/std:codelet",
    ],
)

nv_gxf_cc_library(
    name = "entity_serializer",
    srcs = ["entity_serializer.cpp"],
    hdrs = ["entity_serializer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "endpoint",
        "//gxf/core:component",
    ],
)

nv_gxf_cc_library(
    name = "std_entity_serializer",
    srcs = ["std_entity_serializer.cpp"],
    hdrs = ["std_entity_serializer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "component_serializer",
        "entity_serializer",
        "tid_hash",
    ],
)

nv_gxf_cc_library(
    name = "std_entity_id_serializer",
    srcs = ["std_entity_id_serializer.cpp"],
    hdrs = ["std_entity_id_serializer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "entity_serializer",
    ],
)

nv_gxf_cc_library(
    name = "std_component_serializer",
    srcs = ["std_component_serializer.cpp"],
    hdrs = ["std_component_serializer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "component_serializer",
        "//gxf/cuda:cuda_common",
        "//gxf/std:allocator",
        "//gxf/std:tensor",
        "//gxf/std:timestamp",
    ],
)

nv_gxf_cc_library(
    name = "component_serializer",
    srcs = ["component_serializer.cpp"],
    hdrs = ["component_serializer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "endpoint",
        "tid_hash",
        "//gxf/core:component",
    ],
)

nv_gxf_cc_library(
    name = "file_stream",
    srcs = ["file_stream.cpp"],
    hdrs = ["file_stream.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "endpoint",
    ],
)

nv_gxf_cc_library(
    name = "file",
    srcs = ["file.cpp"],
    hdrs = ["file.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "endpoint",
        "//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "serialization_buffer",
    srcs = ["serialization_buffer.cpp"],
    hdrs = ["serialization_buffer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "endpoint",
        "//gxf/std:allocator",
        "//gxf/std:memory_buffer",
    ],
)

nv_gxf_cc_library(
    name = "endpoint",
    srcs = ["endpoint.cpp"],
    hdrs = ["endpoint.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:component",
        "//gxf/std:allocator"
    ],
)

nv_gxf_cc_library(
    name = "tid_hash",
    hdrs = ["tid_hash.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_c_api",
    ],
)

load("//registry/build:registry.bzl", "register_extension")
exports_files(["LICENSE"])

register_extension(
    name = "register_serialization_ext",
    extension = "serialization",
    uuid = "bc573c2f-89b3-d4b0-8061-2da8b11fe79a",
    version = "2.6.0",
    headers = [
        "BUILD.public",
        "component_serializer.hpp",
        "endpoint.hpp",
        "entity_recorder.hpp",
        "entity_replayer.hpp",
        "entity_serializer.hpp",
        "file.hpp",
        "file_stream.hpp",
        "serialization_buffer.hpp",
        "std_component_serializer.hpp",
        "std_entity_id_serializer.hpp",
        "std_entity_serializer.hpp",
        "tid_hash.hpp",
    ],
    python_sources = [
        "//gxf/serialization/bindings:__init__.py",
        "//gxf/serialization:Components.py",
        "//gxf/serialization:__init__.py",
    ],
    license_file = ":LICENSE",
    url = "www.nvidia.com",
    labels = ["gxf"],
    badges = [""],
    priority = "1",
    local_dependencies = ["//gxf/std:register_std_ext"],
    visibility = ["//visibility:public"],
)
