"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("//gxf:gxf.bzl", "nv_gxf_app", "nv_gxf_pkg", "nv_gxf_test_app", "nv_gxf_sub_graph")


# Create YAML files for test_cpu_burn

test_cpu_burn_yaml_files = [
    "test_cpu_burn_1.yaml",
    "test_cpu_burn_3.yaml",
    "test_cpu_burn_10.yaml",
    "test_cpu_burn_25.yaml",
    "test_cpu_burn_100.yaml",
]

genrule(
    name = "test_cpu_burn_gen",
    outs = test_cpu_burn_yaml_files,
    cmd = "$(location :test_cpu_burn) $(RULEDIR)",
    message = "----- Generating the test_cpu_burn_*.yaml files",
    tools = [":test_cpu_burn"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_cpu_burn",
    srcs = ["test_cpu_burn.py"],
)

test_stress_cpu_gpu_storage_yaml_files = [
    "test_stress_cpu_gpu_storage_gen.yaml",
]

genrule(
    name = "test_stress_cpu_gpu_storage_gen",
    outs = test_stress_cpu_gpu_storage_yaml_files,
    cmd = "$(location :test_stress_cpu_gpu_storage) $(RULEDIR)",
    message = "----- Generating the test_stress_cpu_gpu_storage_gen.yaml file",
    tools = [":test_stress_cpu_gpu_storage"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_cpu_gpu_storage",
    srcs = ["test_stress_cpu_gpu_storage.py"],
)

test_stress_broadcast_gather_yaml_files = [
    "test_stress_broadcast_gather_greedy_gen.yaml",
    "test_stress_broadcast_gather_multithread_gen.yaml",
]

genrule(
    name = "test_stress_broadcast_gather_gen",
    outs = test_stress_broadcast_gather_yaml_files,
    cmd = "$(location :test_stress_broadcast_gather) $(RULEDIR)",
    message = "----- Generating the test_stress_broadcast_gather_gen.yaml file",
    tools = [":test_stress_broadcast_gather"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_broadcast_gather",
    srcs = ["test_stress_broadcast_gather.py"],
)

test_stress_tensor_device_to_host_yaml_files = [
    "test_stress_tensor_device_to_host_gen.yaml",
]

genrule(
    name = "test_stress_tensor_device_to_host_gen",
    outs = test_stress_tensor_device_to_host_yaml_files,
    cmd = "$(location :test_stress_tensor_device_to_host) $(RULEDIR)",
    message = "----- Generating the test_stress_tensor_device_to_host_gen.yaml file",
    tools = [":test_stress_tensor_device_to_host"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_tensor_device_to_host",
    srcs = ["test_stress_tensor_device_to_host.py"],
)

test_stress_ping_yaml_files = [
    "test_stress_ping_gen.yaml",
]

genrule(
    name = "test_stress_ping_gen",
    outs = test_stress_ping_yaml_files,
    cmd = "$(location :test_stress_ping) $(RULEDIR)",
    message = "----- Generating the test_stress_ping_gen.yaml file",
    tools = [":test_stress_ping"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_ping",
    srcs = ["test_stress_ping.py"],
)

test_stress_tensor_host_to_device_yaml_files = [
    "test_stress_tensor_host_to_device_gen.yaml",
]

genrule(
    name = "test_stress_tensor_host_to_device_gen",
    outs = test_stress_tensor_host_to_device_yaml_files,
    cmd = "$(location :test_stress_tensor_host_to_device) $(RULEDIR)",
    message = "----- Generating the test_stress_tensor_host_to_device_gen.yaml file",
    tools = [":test_stress_tensor_host_to_device"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_tensor_host_to_device",
    srcs = ["test_stress_tensor_host_to_device.py"],
)


# Make yaml files available for tests.

yaml_files = glob(["*.yaml"])

filegroup(
    name = "yaml_files",
    srcs = yaml_files + test_cpu_burn_yaml_files,
    visibility = ["//visibility:public"],
)

exports_files(yaml_files)

# Example app using one of the tests

nv_gxf_app(
    name = "test_ping",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_ping.yaml",
    docker_base_image = "//engine/build:docker_base_image",
    docker_repository = "nv-gxf-dev/dev-4",
)

nv_gxf_app(
    name = "test_synchronization_multi",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_synchronization_multi.yaml",
)

nv_gxf_app(
    name = "test_ping_multithread",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_ping_multi_thread.yaml",
)

nv_gxf_app(
    name = "test_ping_multithread_interrupt",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_ping_multi_thread_interrupt.yaml",
)

nv_gxf_app(
    name = "test_multithread_mock_codelet",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_multi_thread_mock_codelet.yaml",
)

nv_gxf_app(
    name = "test_greedy_mock_codelet",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_greedy_mock_codelet.yaml",
)

nv_gxf_app(
    name = "test_entity_group_default",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_entity_group_default.yaml",
)

nv_gxf_app(
    name = "test_entity_group_users",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_entity_group_users.yaml",
)

nv_gxf_app(
    name = "test_entity_group_default_and_users",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_entity_group_default_and_users.yaml",
)

nv_gxf_app(
    name = "test_entity_group_root_and_subgraph",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    data = [
        "entity_group_subgraph.yaml"
    ],
    srcs = ["test_entity_group_root_and_subgraph.yaml"]
)

nv_gxf_sub_graph(
    name = "entity_group_sub_graph",
    sub_graphs = [
        "entity_group_subgraph.yaml"
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_test_app(
    name = "test_entity_group_root_and_subgraph_bazel_target",
    sub_graphs = ["entity_group_sub_graph"],
    srcs = [
        "test_entity_group_root_and_subgraph.yaml",
    ],
    extensions = [
        "//gxf/sample",
    ]
)

nv_gxf_app(
    name = "test_ping_periodic",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_ping_periodic.yaml",
)

nv_gxf_app(
    name = "test_periodic",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_periodic.yaml",
)

nv_gxf_app(
    name = "test_ping_broadcast",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_ping_broadcast.yaml",
)

nv_gxf_app(
    name = "test_ping_split",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    srcs = ["test_ping_graph.yaml", "test_ping_graph_parameters.yaml"],
)


nv_gxf_app(
    name = "test_timestamps",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_timestamps.yaml",
)

nv_gxf_app(
    name = "test_pingrx_async",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_pingrx_async.yaml",
)

nv_gxf_app(
    name = "test_pingtx_async",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_pingtx_async.yaml",
)

nv_gxf_app(
    name = "test_tensor_copier_device_to_system",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_tensor_copier_device_to_system.yaml",
)

nv_gxf_app(
    name = "test_stress_cpu_gpu_storage_app",
    extensions = [
        "//gxf/std",
        "//gxf/npp",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_cpu_gpu_storage_gen.yaml",
)

nv_gxf_app(
    name = "test_stress_broadcast_gather_greedy_app",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/npp",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_broadcast_gather_greedy_gen.yaml",
)


nv_gxf_app(
    name = "test_stress_broadcast_gather_multithread_app",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/npp",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_broadcast_gather_multithread_gen.yaml",
)

nv_gxf_app(
    name = "test_stress_tensor_device_to_host_app",
    extensions = [
        "//gxf/std",
        "//gxf/npp",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_tensor_device_to_host_gen.yaml",
)

nv_gxf_app(
    name = "test_stress_tensor_host_to_device_app",
    extensions = [
        "//gxf/std",
        "//gxf/npp",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_tensor_host_to_device_gen.yaml",
)


nv_gxf_app(
    name = "test_stress_ping_app",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_ping_gen.yaml",
)

nv_gxf_app(
    name = "test_metric",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_metric.yaml",
)

nv_gxf_app(
    name = "hello_world_multi",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_hello_world_multi.yaml",
)

nv_gxf_app(
    name = "hello_world_multi_unscheduled",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_hello_world_multi_unscheduled.yaml",
)

nv_gxf_app(
    name = "hello_world_unscheduled",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_hello_world_unscheduled.yaml",
)

nv_gxf_app(
    name = "hello_world",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_hello_world.yaml",
)

nv_gxf_app(
    name = "test_subgraph_global_parameter",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    data = [
        "test_subgraph_global_parameter_subgraph.yaml"
    ],
    gxf_file = "test_subgraph_global_parameter_app.yaml",
)

nv_gxf_app(
    name = "test_delayed_periodic",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_delayed_periodic.yaml",
)

nv_gxf_app(
    name = "test_delayed_periodic_stress",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_delayed_periodic_stress.yaml",
)

nv_gxf_app(
    name = "test_downstream_scheduling",
    extensions = [
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_downstream_scheduling.yaml",
)

py_binary(
    name = "test_stress_subgraph",
    srcs = ["test_stress_subgraph.py"],
)

test_stress_subgraph_yaml_files = [
    "test_stress_subgraph_gen.yaml",
]

genrule(
    name = "test_stress_subgraph_gen",
    outs = test_stress_subgraph_yaml_files,
    cmd = "$(location :test_stress_subgraph) $(RULEDIR)",
    message = "----- Generating the test_stress_subgraph_gen.yaml file",
    tools = [":test_stress_subgraph"],
    visibility = ["//visibility:public"],
)

nv_gxf_app(
    name = "test_stress_subgraph_app",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    data = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
    ],
    gxf_file = "test_stress_subgraph_gen.yaml"
)

nv_gxf_app(
    name = "test_subgraph_split_1",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    data = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
        "gather_subgraph.param.yaml"
    ],
    srcs = ["test_subgraph_split.yaml", "test_subgraph_split.param1.yaml"]
)

nv_gxf_app(
    name = "test_subgraph_split_2",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    data = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
        "gather_subgraph.param.yaml"
    ],
    srcs = ["test_subgraph_split.yaml", "test_subgraph_split.param2.yaml"]
)

nv_gxf_sub_graph(
    name = "gather_forward_sub_graph",
    sub_graphs = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
        "gather_subgraph.param.yaml"
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_sub_graph(
    name = "forward_sub_graph",
    sub_graphs = [
        "forward_subgraph.yaml",
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_sub_graph(
    name = "gather_sub_graph",
    sub_graphs = [
        "gather_subgraph.yaml",
        "gather_subgraph.param.yaml"
    ],
    extensions = [
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_test_app(
    name = "test_subgraph_bazel_target",
    sub_graphs = ["gather_forward_sub_graph"],
    srcs = [
        "test_subgraph_split.yaml",
        "test_subgraph_split.param2.yaml",
    ],
    extensions = [
        "//gxf/sample",
    ]
)

nv_gxf_test_app(
    name = "test_subgraph_bazel_target_multi",
    sub_graphs = ["forward_sub_graph", "gather_sub_graph"],
    srcs = [
        "test_subgraph_split.yaml",
        "test_subgraph_split.param2.yaml",
    ],
    extensions = [
        "//gxf/sample",
    ]
)

nv_gxf_app(
    name = "test_multi_thread_pool",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_multi_thread_pool.yaml",
)

nv_gxf_app(
    name = "test_ping_target_time_multi_thread",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
        "//gxf/multimedia"
    ],
    gxf_file = "test_ping_target_time_multi_thread.yaml",
)

nv_gxf_app(
    name = "test_message_available_frequency_throttler",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_message_available_frequency_throttler.yaml",
)

nv_gxf_app(
    name = "test_multi_thread_exit",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_multi_thread_exit.yaml",
)

nv_gxf_app(
    name = "test_deadlock_timeout_multi_thread",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_deadlock_timeout_multi_thread.yaml",
)

nv_gxf_app(
    name = "test_deadlock_timeout_greedy",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_deadlock_timeout_greedy.yaml",
)

nv_gxf_test_app(
    name= "test_ping_topics",
    srcs = ["test_ping_topics.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name= "test_ping_multi_connect",
    srcs = ["test_ping_multi_connect.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_ebs_mock_codelet",
    srcs = ["test_ebs_mock_codelet.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_entity_group_default_ebs",
    srcs = ["test_entity_group_default_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_entity_group_users_ebs",
    srcs = ["test_entity_group_users_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_entity_group_default_and_users_ebs",
    srcs = ["test_entity_group_default_and_users_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_entity_group_root_and_subgraph_ebs",
    srcs = ["test_entity_group_root_and_subgraph_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    data = [
        "entity_group_subgraph.yaml"
    ],
)

nv_gxf_test_app(
    name = "hello_world_ebs",
    srcs = ["test_hello_world_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "hello_world_ebs_unscheduled",
    srcs = ["test_hello_world_ebs_unscheduled.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_ebs_pool",
    srcs = ["test_ebs_pool.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)


nv_gxf_test_app(
    name = "test_ping_target_time_ebs",
    srcs = ["test_ping_target_time_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
        "//gxf/multimedia"
    ],
)

nv_gxf_test_app(
    name = "test_message_available_frequency_throttler_ebs",
    srcs = ["test_message_available_frequency_throttler_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_deadlock_timeout_ebs",
    srcs = ["test_deadlock_timeout_ebs.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_or_scheduling_term",
    srcs = ["test_ping_or_scheduling_term.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)

nv_gxf_test_app(
    name = "test_or_scheduling_term_greedy",
    srcs = ["test_ping_or_scheduling_term_greedy.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
)
