"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
load(
    "@com_nvidia_gxf//gxf:gxf.bzl",
    "nv_gxf_app",
    "nv_gxf_test_app",
    "nv_gxf_multi_app",
    "nv_gxf_multi_test_app",
)
load("//gxf/python:pygxf.bzl", "nv_pygxf_test")

exports_files(["LICENSE"])

nv_gxf_app(
    name = "ucx_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_ping_ucx_tx.yaml" ],
)

nv_gxf_app(
    name = "ucx_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = [
        "test_ping_ucx_rx.yaml",
    ],
)


nv_gxf_multi_app(
    name = "test_ucx_ping",
    apps = [
        "ucx_tx",
        "ucx_rx",
    ],
    tags = ["exclusive", "manual"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "ucx_multi_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_ping_multiple_ucx_tx.yaml" ],
)


nv_gxf_app(
    name = "ucx_multi_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = [
        "test_ping_multiple_ucx_rx.yaml",
    ],
)

nv_gxf_multi_app(
    name = "test_ucx_multi_ping",
    apps = [
        "ucx_multi_tx",
        "ucx_multi_rx",
    ],
    tags = ["exclusive", "manual"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "ucx_tensor_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_tx.yaml" ],
)

nv_gxf_app(
    name = "ucx_tensor_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_rx.yaml" ],
)

nv_gxf_multi_test_app(
    name = "test_ucx_tensor",
    apps = [
        "ucx_tensor_tx",
        "ucx_tensor_rx",
    ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "ucx_tensor_rx_cpu_only",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_rx_cpu_only.yaml" ],
)

nv_gxf_app(
    name = "ucx_tensor_rx_EBS",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_rx_EBS.yaml" ],
)

nv_gxf_app(
    name = "ucx_tensor_tx_EBS",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_tx_EBS.yaml" ],
)

nv_gxf_multi_test_app(
    name = "test_ucx_tensor_EBS",
    apps = [
        "ucx_tensor_tx_EBS",
        "ucx_tensor_rx_EBS",
    ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "ucx_tensor_rx_EBS_sync",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_rx_EBS_sync.yaml" ],
)

nv_gxf_app(
    name = "ucx_tensor_tx_EBS_sync",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_tx_EBS_sync.yaml" ],
)

nv_gxf_multi_test_app(
    name = "test_ucx_tensor_EBS_sync",
    apps = [
        "ucx_tensor_tx_EBS_sync",
        "ucx_tensor_rx_EBS_sync",
    ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "ucx_tensor_rx_multiple_entity",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_rx_multiple_entity.yaml" ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "ucx_tensor_tx_multiple_1",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_tx_multiple1.yaml" ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "ucx_tensor_tx_multiple_2",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_tensor_ucx_tx_multiple2.yaml" ],
    tags = ["exclusive", "manual"],
)

nv_gxf_multi_test_app(
    name = "test_ucx_multiple_tx_rx",
    apps = [
        "ucx_tensor_rx_multiple_entity",
        "ucx_tensor_tx_multiple_1",
        "ucx_tensor_tx_multiple_2",
    ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "ucx_forward_tx_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_forward_tx_rx.yaml" ],
)

nv_gxf_app(
    name = "ucx_forward_tx_rx_sync",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_forward_tx_rx_sync.yaml" ],
)

nv_gxf_app(
    name = "ucx_multiple_tensor_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multiple_tensor_tx.yaml" ],
)

nv_gxf_app(
    name = "ucx_multiple_tensor_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multiple_tensor_rx.yaml" ],
)

nv_gxf_multi_test_app(
    name = "test_ucx_multiple_tensor",
    apps = [
        "ucx_multiple_tensor_tx",
        "ucx_forward_tx_rx",
        "ucx_multiple_tensor_rx",
    ],
    tags = ["exclusive", "manual"],
)

nv_pygxf_test(
    name = "test_python_ucx_rx",
    srcs = ["test_python_ucx_rx.py"],
    extensions = [
        "//gxf/cuda",
        "//gxf/multimedia",
        "//gxf/python_codelet",
        "//gxf/sample",
        "//gxf/serialization",
        "//gxf/std",
        "//gxf/ucx",
    ],
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/core:core_pybind",
        "//gxf/cuda:cuda_extension",
        "//gxf/multimedia:multimedia_extension",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/sample:sample_extension",
        "//gxf/serialization:serialization_extension",
        "//gxf/std:std_extension",
        "//gxf/ucx:ucx_extension",
    ],
)

nv_pygxf_test(
    name = "test_python_ucx_tx",
    srcs = ["test_python_ucx_tx.py"],
    extensions = [
        "//gxf/cuda",
        "//gxf/multimedia",
        "//gxf/python_codelet",
        "//gxf/sample",
        "//gxf/serialization",
        "//gxf/std",
        "//gxf/ucx",
    ],
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/core:core_pybind",
        "//gxf/cuda:cuda_extension",
        "//gxf/multimedia:multimedia_extension",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/sample:sample_extension",
        "//gxf/serialization:serialization_extension",
        "//gxf/std:std_extension",
        "//gxf/ucx:ucx_extension",
    ],
)

nv_gxf_multi_test_app(
    name = "test_python_ucx",
    apps = [
        "test_python_ucx_rx",
        "test_python_ucx_tx",
    ],
    tags = ["exclusive", "manual"],
)