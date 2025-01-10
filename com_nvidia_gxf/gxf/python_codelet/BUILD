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
    # "nv_gxf_components_pyclass_gen"
)

load("//gxf:pybind.bzl", "pybind_library")

# nv_gxf_components_pyclass_gen(
#     name = "comp_gen",
#     extensions = [
#         "//gxf/python_codelet",
#     ],
# )

exports_files([
    "__init__.py",
    "codelet.py"
])

py_library(
    name = "python_codelet_extension",
    srcs = [
        "__init__.py",
        "Components.py",
        "codelet.py",
        "py_compute_entity.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/python_codelet:pycodelet",
        "//gxf/python_codelet:pydata_pybind",
        "//gxf/core:core_pybind",
        "//gxf/std:tensor_pybind",
        "//gxf/std:vault_pybind"
    ],
)

nv_gxf_cc_library(
    name = "pydata",
    hdrs = ["pydata.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@pybind11",
        "//third_party:python",
        "//gxf/std",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "py_codelet",
    srcs = ["py_codelet.cpp"],
    hdrs = ["py_codelet.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:parameter_parser_std",
        "//gxf/cuda:cuda_stream_id",
        "//gxf/cuda:cuda_stream_pool",
        "//gxf/std:allocator",
        "//gxf/std:clock",
        "//gxf/std:codelet",
        "//gxf/std:scheduling_terms",
        "//gxf/std:receiver",
        "//gxf/std:transmitter",
        "//gxf/std:extension_factory_helper",
        "@pybind11",
        "//third_party:python",
    ],
)

nv_gxf_cc_extension(
    name = "python_codelet",
    srcs = ["python_codelet.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf",
        "py_codelet",
        "pydata",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/cuda",
    ],
)

pybind_library(
    name = "pycodelet",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/python_codelet",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/pycodelet.cpp",
    ],
)

pybind_library(
    name = "pydata_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/python_codelet:pydata"
    ],
    cc_srcs = [
        "bindings/pydata.cpp",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_python_codelet_ext",
    extension = "python_codelet",
    headers = [
        "BUILD.public",
        "py_codelet.hpp",
        "pydata.hpp",
        ],
    labels = ["gxf"],
    python_bindings = [
        "//gxf/python_codelet:pycodelet.so",
        "//gxf/python_codelet:pydata_pybind.so"
    ],
    python_sources = [
        "//gxf/python_codelet:__init__.py",
        "//gxf/python_codelet:Components.py",
        "//gxf/python_codelet:codelet.py",
        "//gxf/python_codelet:py_compute_entity.py"],
    badges = [""],
    license_file = ":LICENSE",
    priority = "1",
    url = "www.nvidia.com",
    uuid = "787daddc-1c34-11ec-9621-0242ac130002",
    version = "0.6.0",
    local_dependencies = ["//gxf/std:register_std_ext",
                          "//gxf/cuda:register_cuda_ext"],
    compute_dependencies = ["cuda"],
    visibility = ["//visibility:public"],
)
