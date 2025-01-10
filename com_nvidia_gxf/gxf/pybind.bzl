"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# A custom bazel rule to compile a C++ Python binding using pybind11. The rule creates a target
# which can be used in py_library and py_binary rules as a dependency to load the Python
# binding. The same target can also be used as a dependency in another pybind_library.
def pybind_library(name, cc_hdrs = [], cc_srcs = [], cc_deps = [], py_srcs = [], py_deps = [], deps = [], data = []):
    so_name = name + ".so"
    lib_name = name + "_pybind"

    filtered_deps = []
    for x in deps:
        if len(x.split(":")) > 1:
            filtered_deps += [x + "_pybind"]
        else:
            keys = x.split("/")
            key = keys[len(keys) - 1]
            filtered_deps += [x + ":" + key + "_pybind"]
    native.cc_library(
        name = lib_name,
        hdrs = cc_hdrs,
        srcs = cc_srcs,
        alwayslink = True,
        visibility = ["//visibility:public"],
        copts = ["-Wno-unused-function"],
        # Provides dependency of Python2.7@TX2 when compiling for AARCH64.
        # Defaults to host Python2.
        deps = cc_deps + filtered_deps + ["@pybind11"] + ["@com_nvidia_gxf//third_party:python"],
    )
    native.cc_binary(
        name = so_name,
        visibility = ["//visibility:public"],
        linkopts = [],
        linkshared = 1,
        linkstatic = 1,
        deps = [lib_name],
    )
    native.py_library(
        name = name,
        visibility = ["//visibility:public"],
        srcs = py_srcs,
        deps = py_deps,
        data = data + [":" + so_name],
        imports = [":" + so_name],
    )
