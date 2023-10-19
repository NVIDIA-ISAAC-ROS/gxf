"""
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "//gxf:gxf.bzl",
    "nv_gxf_c_library",
    "nv_gxf_cc_library",
)

load("//gxf:pybind.bzl", "pybind_library")
load("@com_nvidia_gxf//packaging:packaging.bzl", "nv_gxf_pkg_library")

exports_files(["py_module_builder.py"])

nv_gxf_cc_library(
    name = "component",
    hdrs = [
        "component.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "entity",
        "expected",
        "gxf_c_api",
        "handle",
        "parameter",
        "registrar",
        "registry",
        "//common",
        "//gxf/std:parameter_storage",
    ],
)

nv_gxf_cc_library(
    name = "entity",
    hdrs = [
        "entity.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "handle",
        "registry",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "expected",
    hdrs = [
        "expected.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "gxf_c_api",
        "//common",
    ],
)

nv_gxf_c_library(
    name = "gxf_c_api",
    hdrs = [
        "gxf.h",
        "gxf_ext.h",
    ],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "handle",
    hdrs = [
        "handle.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "registry",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "parameter",
    hdrs = [
        "parameter.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "handle",
        "//common",
        "//gxf/std:parameter_parser",
        "//gxf/std:parameter_wrapper",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "resource",
    hdrs = [
        "resource.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "handle",
    ],
)

nv_gxf_cc_library(
    name = "registry",
    hdrs = ["registry.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:type_name",
    ],
)

nv_gxf_cc_library(
    name = "registrar",
    hdrs = [
        "registrar.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "resource",
        "//common",
        "//gxf/std:parameter_registrar",
        "//gxf/std:parameter_storage",
        "//gxf/std:resource_manager",
        "//gxf/std:resource_registrar",
    ],
)

nv_gxf_cc_library(
    name = "gxf_cpp_api",
    visibility = ["//visibility:public"],
    deps = [
        "component",
        "entity",
        "expected",
        "gxf_c_api",
        "handle",
        "parameter",
        "registry",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "gxf",
    srcs = [
        "gxf.cpp",
        "runtime.cpp",
    ],
    hdrs = [
        "runtime.hpp",
    ],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "gxf_c_api",
        "gxf_cpp_api",
        "//common",
        "//gxf/std:entity_executor",
        "//gxf/std:entity_warden",
        "//gxf/std:extension_loader",
        "//gxf/std:parameter_storage",
        "//gxf/std:program",
        "//gxf/std:resource_manager",
        "//gxf/std:resource_registrar",
        "//gxf/std:system",
        "//gxf/std:type_registry",
        "//gxf/std:yaml_file_loader",
        "@yaml-cpp",
    ],
    alwayslink = True,
)

py_binary(
    name="py_module_builder",
    srcs=[
        "py_module_builder.py",
    ],
    deps=[
        "core_pybind",
    ],
    data=[
        "//gxf/core:gxf",
    ],
    visibility=["//visibility:public"],
)

pybind_library(
    name = "core_pybind",
    cc_deps = [
        "gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/core.cpp",
    ],
    py_srcs = [
        "__init__.py",
        "Gxf.py",
        "logger.py"
    ],
)

cc_binary(
    name = "libgxf_core.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = ["gxf"],
    linkopts = [
        "-Wl,--exclude-libs,libyaml_file_loader.lo",
        "-Wl,--exclude-libs,libyaml-cpp.a"
    ],
)

cc_import(
    name = "libgxf_core",
    hdrs = [
        "gxf.h",
        "gxf_ext.h",
    ],
    shared_library = "libgxf_core.so",
    visibility = ["//visibility:public"],
)

nv_gxf_pkg_library(
    name = "core",
    build = "BUILD.release",
    headers = [
        "component.hpp",
        "entity.hpp",
        "expected.hpp",
        "gxf.h",
        "gxf_ext.h",
        "handle.hpp",
        "parameter.hpp",
        "registrar.hpp",
    ],
    binaries = [
    ":core_pybind.so",
    ":libgxf_core.so",
    ":gxf",
    ]
)
