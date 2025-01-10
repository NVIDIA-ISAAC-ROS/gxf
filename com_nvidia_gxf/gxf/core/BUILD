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
    "nv_gxf_c_library",
    "nv_gxf_cc_library",
    "nv_gxf_cc_binary"
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
        "parameter_storage",
        "registrar",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "entity",
    hdrs = [
        "entity.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "entity_item",
        "expected",
        "gxf_c_api",
        "handle",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "entity_item",
    hdrs = [
        "entity_item.hpp"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "gxf_c_api",
    ],
)

nv_gxf_cc_library(
    name = "expected",
    hdrs = [
        "common_expected_macro.hpp",
        "expected.hpp",
        "expected_macro.hpp"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "gxf_c_api",
        "//common",
        "@magic_enum",
    ],
)

nv_gxf_cc_library(
    name = "filepath",
    hdrs = ["filepath.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@yaml-cpp",
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
        "parameter_parser",
        "parameter_wrapper",
        "//common",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "parameter_parser",
    hdrs = [
        "parameter_parser.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "expected",
        "filepath",
        "gxf_c_api",
        "handle",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "parameter_parser_std",
    hdrs = ["parameter_parser_std.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "handle",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "parameter_registrar",
    srcs = ["parameter_registrar.cpp"],
    hdrs = ["parameter_registrar.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "expected",
        "gxf_c_api",
        "parameter_storage",
        "type_registry",
    ],
)

nv_gxf_cc_library(
    name = "parameter_storage",
    srcs = ["parameter_storage.cpp"],
    hdrs = ["parameter_storage.hpp"],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "parameter",
    ],
)

nv_gxf_cc_library(
    name = "parameter_wrapper",
    hdrs = [
        "parameter_wrapper.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "expected",
        "gxf_c_api",
        "handle",
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
    name = "registrar",
    hdrs = [
        "registrar.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "parameter_registrar",
        "parameter_storage",
        "resource",
        "resource_manager",
        "resource_registrar",
        "//common",
    ],
)

nv_gxf_cc_library(
    name = "resource_manager",
    srcs = ["resource_manager.cpp"],
    hdrs = ["resource_manager.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "gxf_c_api",
        "handle",
    ],
)

nv_gxf_cc_library(
    name = "resource_registrar",
    hdrs = ["resource_registrar.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "gxf_c_api",
    ],
)

nv_gxf_cc_library(
    name = "type_registry",
    srcs = ["type_registry.cpp"],
    hdrs = ["type_registry.hpp"],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "expected",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "gxf_cpp_api",
    visibility = ["//visibility:public"],
    deps = [
        "component",
        "entity",
        "expected",
        "filepath",
        "gxf_c_api",
        "handle",
        "parameter",
        "parameter_parser",
        "parameter_parser_std",
        "parameter_registrar",
        "parameter_storage",
        "parameter_wrapper",
        "resource_manager",
        "resource_registrar",
        "type_registry",
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
        "parameter_storage",
        "resource_manager",
        "resource_registrar",
        "type_registry",
        "//common",
        "//gxf/std:entity_executor",
        "//gxf/std:entity_warden",
        "//gxf/std:extension_loader",
        "//gxf/std:program",
        "//gxf/std:system",
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

nv_gxf_cc_binary(
    name = "libgxf_core.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = ["gxf"],
    linkopts = [
        "-Wl,--exclude-libs,libyaml_file_loader.lo",
        "-Wl,--exclude-libs,libyaml-cpp.a",
        "-Wl,-soname,libgxf_core.so",
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
        "filepath.hpp",
        "gxf.h",
        "gxf_ext.h",
        "handle.hpp",
        "parameter.hpp",
        "parameter_parser.hpp",
        "parameter_parser_std.hpp",
        "parameter_storage.hpp",
        "parameter_wrapper.hpp",
        "registrar.hpp",
        "resource_manager.hpp",
        "resource_registrar.hpp",
        "type_registry.hpp",
    ],
    binaries = [
    ":core_pybind.so",
    ":libgxf_core.so",
    ":gxf",
    ]
)
