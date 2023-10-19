"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "//gxf:gxf.bzl",
    "nv_gxf_cc_test_group",
)

cc_library(
    name = "test_main",
    srcs = ["test_main.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//external:gflags",
        "//common",
        "//gxf/core:gxf",
        "//gxf/core:libgxf_core",
        "@gtest",
        "@breakpad//:breakpad"
    ]
)

nv_gxf_cc_test_group(
    name = "all_test",
    apps = [
        "gxf/test/apps/test_load_order.yaml",
        "gxf/test/apps/example_basics_1.yaml",
        "gxf/test/apps/example_basics_2.yaml",
        "gxf/test/apps/example_basics_3.yaml",
        "gxf/test/apps/test_cpu_burn_1.yaml",
        "gxf/test/apps/test_cpu_burn_10.yaml",
        "gxf/test/apps/test_cpu_burn_100.yaml",
        "gxf/test/apps/test_cpu_burn_25.yaml",
        "gxf/test/apps/test_cpu_burn_3.yaml",
        "gxf/test/apps/test_factorial.yaml",
        "gxf/test/apps/test_fixed_vector_parameter.yaml",
        "gxf/test/apps/test_handle_parameter.yaml",
        "gxf/test/apps/test_hello_world_multi_unscheduled.yaml",
        "gxf/test/apps/test_hello_world_multi.yaml",
        "gxf/test/apps/test_hello_world_unscheduled.yaml",
        "gxf/test/apps/test_hello_world.yaml",
        "gxf/test/apps/test_load_parameter_from_yaml.yaml",
        "gxf/test/apps/test_periodic.yaml",
        "gxf/test/apps/test_ping_target_time.yaml",
        "gxf/test/apps/test_ping_target_time_multi_thread.yaml",
        "gxf/test/apps/test_ping.yaml",
        "gxf/test/apps/test_ping_composer.yaml",
        "gxf/test/apps/test_ping_log.yaml",
        "gxf/test/apps/test_ping_multi_thread.yaml",
        "gxf/test/apps/test_ping_3.yaml",
        "gxf/test/apps/test_ping_broadcast.yaml",
        "gxf/test/apps/test_ping_gather.yaml",
        "gxf/test/apps/test_ping_gather_async.yaml",
        "gxf/test/apps/test_ping_periodic.yaml",
        "gxf/test/apps/test_pingrx_async_greedy.yaml",
        "gxf/test/apps/test_pingrx_async.yaml",
        "gxf/test/apps/test_pingtx_async_2_gather_greedy.yaml",
        "gxf/test/apps/test_pingtx_async_2_gather.yaml",
        "gxf/test/apps/test_pingtx_async_greedy.yaml",
        "gxf/test/apps/test_pingtx_async.yaml",
        "gxf/test/apps/test_register_parameter_interface.yaml",
        "gxf/test/apps/test_subgraph.yaml",
        "gxf/test/apps/test_subgraph_global_parameter_app.yaml",
        "gxf/test/apps/test_std_parameter.yaml",
        "gxf/test/apps/test_step_count.yaml",
        "gxf/test/apps/test_synchronization.yaml",
        "gxf/test/apps/test_synchronization_threshold.yaml",
        "gxf/test/apps/test_tensor_copier_device_to_host.yaml",
        "gxf/test/apps/test_tensor_copier_device_to_system.yaml",
        "gxf/test/apps/test_tensor_copier_host_to_device.yaml",
        "gxf/test/apps/test_tensor_copier_host_to_system.yaml",
        "gxf/test/apps/test_tensor_copier_system_to_host.yaml",
        "gxf/test/apps/test_tensor_copier_system_to_device.yaml",
        "gxf/test/apps/test_tensor_stride.yaml",
        "gxf/test/apps/test_timed_throttler.yaml",
        "gxf/test/apps/test_round_robin_broadcast.yaml",
        "gxf/test/apps/test_delayed_periodic.yaml",
        "gxf/test/apps/test_downstream_scheduling.yaml",
    ],
    data = ["//gxf/test/apps:yaml_files"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/test/",
    deps = [
        "test_main",
        "//gxf/core:gxf",
        "//gxf/std:yaml_file_loader",
        "@yaml-cpp"
    ],
)
