"""
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_cc_extension", "nv_gxf_cc_library")
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_extension")
load("@com_nvidia_gxf//registry/build:registry.bzl", "register_extension")

# load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_components_pyclass_gen")

# nv_gxf_components_pyclass_gen(
#     name="pygen",
#     extensions=[
#         "ucx",
#     ],
# )

nv_gxf_cc_extension(
    name = "ucx",
    srcs = ["ucx_extension.cpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "ucx_context",
        "ucx_receiver",
        "ucx_transmitter",
    ],
    ext_deps = [
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
    ],
)

nv_gxf_cc_library(
    name = "ucx_context",
    srcs = ["ucx_context.cpp"],
    hdrs = ["ucx_context.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_gxf//gxf/std",
        "ucx_receiver",
        "ucx_transmitter",
        "//third_party/ucx:ucx",
        "ucx_entity_serializer"
    ],
)

nv_gxf_cc_library(
    name = "ucx_common",
    srcs = ["ucx_common.cpp"],
    hdrs = ["ucx_common.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/ucx:ucx",
        "@com_nvidia_gxf//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "ucx_receiver",
    srcs = ["ucx_receiver.cpp"],
    hdrs = ["ucx_receiver.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_gxf//gxf/std",
        "//third_party/ucx:ucx",
        "ucx_common",
        "ucx_entity_serializer",
        "@com_nvidia_gxf//third_party:cudart"
    ],
)

nv_gxf_cc_library(
    name = "ucx_transmitter",
    srcs = ["ucx_transmitter.cpp"],
    hdrs = ["ucx_transmitter.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_gxf//gxf/std",
        "//third_party/ucx:ucx",
        "ucx_common",
        "ucx_entity_serializer"
    ],
)

nv_gxf_cc_library(
    name = "ucx_component_serializer",
    srcs = ["ucx_component_serializer.cpp"],
    hdrs = ["ucx_component_serializer.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "ucx_serialization_buffer",
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "ucx_entity_serializer",
    srcs = ["ucx_entity_serializer.cpp"],
    hdrs = ["ucx_entity_serializer.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "ucx_component_serializer",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/std",
    ],
)

nv_gxf_cc_library(
    name = "ucx_serialization_buffer",
    srcs = ["ucx_serialization_buffer.cpp"],
    hdrs = ["ucx_serialization_buffer.hpp"],
    copts = [
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/ucx:ucx",
        "ucx_common",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/std",
    ],
)

py_library(
    name = "ucx_extension",
    srcs = [
        "__init__.py",
        "Components.py"
    ],
    visibility = ["//visibility:public"],
    data = [
        "//gxf/ucx:Components.py",
    ],
)

# exports_files(["LICENSE"])

register_extension(
    name = "register_ucx_ext",
    extension = "ucx",
    headers = [ "BUILD.public",
                "ucx_receiver.hpp",
                "ucx_transmitter.hpp",
                "ucx_context.hpp",
                "ucx_common.hpp",
                "ucx_component_serializer.hpp",
                "ucx_entity_serializer.hpp"],
    python_sources = [
            "//gxf/ucx:Components.py",
            "//gxf/ucx:__init__.py",
        ],
    labels = ["gxf"],
    badges = [""],
    license = "NVIDIA",
    license_file = ":LICENSE",
    priority = "1",
    url = "www.nvidia.com",
    uuid = "525f8a1a-dfb5-426b-8ddb-00c3ac839994",
    version = "0.8.0",
    local_dependencies = ["@com_nvidia_gxf//gxf/std:register_std_ext",
                          "@com_nvidia_gxf//gxf/serialization:register_serialization_ext",
                          "@com_nvidia_gxf//gxf/multimedia:register_multimedia_ext"],
    compute_dependencies = ["cuda"],
    visibility = ["//visibility:public"],
)


nv_gxf_pkg_extension (
    name = "ucx",
    extension = "ucx",
    build = "BUILD.release",
    license = "LICENSE",
    headers = [ "ucx_receiver.hpp",
                "ucx_transmitter.hpp",
                "ucx_context.hpp",
                "ucx_common.hpp",
                "ucx_component_serializer.hpp",
                "ucx_entity_serializer.hpp"],
)
