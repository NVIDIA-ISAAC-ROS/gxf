"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "nv_gxf_multi_test_app"
)
exports_files(["LICENSE"])


nv_gxf_app(
    name = "test_single_tx_multi_tensor1",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_single_tx_multi_tensor1.yaml" ],
    tags = ["manual"],
)

nv_gxf_app(
    name = "test_single_tx_multi_tensor2",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_single_tx_multi_tensor2.yaml" ],
    tags = ["manual"],
)

nv_gxf_app(
    name = "test_multi_tensor_greedy_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multi_tensor_greedy_rx.yaml" ],
    tags = ["manual"],
)

nv_gxf_app(
    name = "test_multi_tensor_greedy_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multi_tensor_greedy_tx.yaml" ],
    tags = ["manual"],
)

nv_gxf_multi_test_app(
    name = "test_multi_tensor_greedy",
    apps = [
        "test_multi_tensor_greedy_tx",
        "test_multi_tensor_greedy_rx",
    ],
    tags = ["exclusive", "manual"],
)

nv_gxf_app(
    name = "test_multi_tensor_EBS_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multi_tensor_EBS_rx.yaml" ],
    tags = ["manual"],
)

nv_gxf_app(
    name = "test_multi_tensor_EBS_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multi_tensor_EBS_tx.yaml" ],
    tags = ["manual"],
)

nv_gxf_multi_test_app(
    name = "test_multi_tensor_EBS",
    apps = [
        "test_multi_tensor_EBS_tx",
        "test_multi_tensor_EBS_rx",
    ],
    tags = ["exclusive", "manual"],
)


nv_gxf_app(
    name = "test_multi_tensor_MT_rx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multi_tensor_MT_rx.yaml" ],
    tags = ["manual"],
)


nv_gxf_app(
    name = "test_multi_tensor_MT_tx",
    extensions = [
        "@com_nvidia_gxf//gxf/multimedia",
        "@com_nvidia_gxf//gxf/std",
        "@com_nvidia_gxf//gxf/serialization",
        "@com_nvidia_gxf//gxf/sample",
        "@com_nvidia_gxf//gxf/test/extensions:test",
        "@com_nvidia_gxf//gxf/ucx",

    ],
    srcs = ["test_multi_tensor_MT_tx.yaml" ],
    tags = ["manual"],
)

nv_gxf_multi_test_app(
    name = "test_multi_tensor_MT",
    apps = [
        "test_multi_tensor_MT_tx",
        "test_multi_tensor_MT_rx",
    ],
    tags = ["exclusive", "manual"],
)
