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
    "nv_gxf_cc_extension",
    "nv_gxf_cc_library",
    # "nv_gxf_components_pyclass_gen"
)
load("//gxf:pybind.bzl", "pybind_library")

exports_files([
    "__init__.py",
])

# nv_gxf_components_pyclass_gen(
#     name = "comp_gen",
#     extensions = [
#         "//gxf/std",
#     ],
# )

pybind_library(
    name = "vault_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/vault.cpp",
    ],
)

pybind_library(
    name = "allocator_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/allocator.cpp",
    ],
)

pybind_library(
    name = "clock_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/clock.cpp",
    ],
)

pybind_library(
    name = "receiver_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/receiver.cpp",
    ],
)

pybind_library(
    name = "tensor_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/tensor.cpp",
    ],
)

pybind_library(
    name = "scheduling_terms_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/scheduling_terms.cpp",
    ],
)

pybind_library(
    name = "timestamp_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/timestamp.cpp",
    ],
)

pybind_library(
    name = "eos_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/eos.cpp",
    ],
)

pybind_library(
    name = "transmitter_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/transmitter.cpp",
    ],
)

pybind_library(
    name = "scheduling_condition_pybind",
    cc_deps = [
        "//gxf/core:gxf",
        "//gxf/std",
    ],
    cc_srcs = [
        "bindings/scheduling_condition.cpp",
    ],
)

py_library(
    name = "std_extension",
    srcs = [
        "__init__.py",
        "standard_graph.py",
        "standard_methods.py",
        "compute_entity.py",
        "Components.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:core_pybind",
        "//gxf/std:allocator_pybind",
        "//gxf/std:clock_pybind",
        "//gxf/std:eos_pybind",
        "//gxf/std:receiver_pybind",
        "//gxf/std:scheduling_condition_pybind",
        "//gxf/std:scheduling_terms_pybind",
        "//gxf/std:tensor_pybind",
        "//gxf/std:timestamp_pybind",
        "//gxf/std:transmitter_pybind",
        "//gxf/std:vault_pybind",
    ],
)

nv_gxf_cc_library(
    name = "std_lib",
    hdrs = [
        "epoch_scheduler.hpp",
        "greedy_scheduler.hpp",
        "scheduling_term.hpp",
        "scheduling_term_combiner.hpp",
        "scheduling_terms.hpp",
    ],
    deps = [
        "allocator",
        "async_buffer_receiver",
        "async_buffer_transmitter",
        "block_memory_pool",
        "broadcast",
        "clock",
        "codelet",
        "complex",
        "component_allocator",
        "component_factory",
        "connection",
        "controller",
        "cpu_thread",
        "default_extension",
        "double_buffer_receiver",
        "double_buffer_transmitter",
        "entity_executor",
        "entity_resource_helper",
        "eos",
        "epoch_scheduler",
        "extension",
        "extension_factory_helper",
        "extension_loader",
        "forward",
        "gather",
        "graph_driver",
        "graph_worker",
        "greedy_scheduler",
        "ipc_client",
        "ipc_server",
        "job_statistics",
        "memory_buffer",
        "message_router",
        "metric",
        "monitor",
        "multi_thread_scheduler",
        "event_based_scheduler",
        "network_context",
        "network_router",
        "new_component_allocator",
        "program",
        "queue",
        "receiver",
        "resources",
        "router",
        "router_group",
        "scheduler",
        "subgraph",
        "synchronization",
        "synthetic_clock",
        "system_group",
        "tensor",
        "tensor_copier",
        "timed_throttler",
        "timestamp",
        "topic",
        "transmitter",
        "typed_message_view",
        "unbounded_allocator",
        "vault",
        "yaml_file_loader",
        "//gxf/core:gxf",
    ],
)

nv_gxf_cc_extension(
    name = "std",
    srcs = ["std.cpp"],
    deps = [
        "std_lib",
    ],
)

nv_gxf_cc_library(
    name = "gxf_std_static",
    srcs = ["std.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "std_lib",
    ],
)

nv_gxf_cc_library(
    name = "allocator",
    srcs = ["allocator.cpp"],
    hdrs = ["allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "ipc_server",
    hdrs = ["ipc_server.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "ipc_client",
    hdrs = ["ipc_client.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "async_buffer_receiver",
    srcs = ["async_buffer_receiver.cpp"],
    hdrs = ["async_buffer_receiver.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "receiver",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "async_buffer_transmitter",
    srcs = ["async_buffer_transmitter.cpp"],
    hdrs = ["async_buffer_transmitter.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "transmitter",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "broadcast",
    srcs = ["broadcast.cpp"],
    hdrs = ["broadcast.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
    ],
)

nv_gxf_cc_library(
    name = "block_memory_pool",
    srcs = ["block_memory_pool.cpp"],
    hdrs = ["block_memory_pool.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "allocator",
        "resources",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/pool:fixed_pool_uint64",
        "//third_party:cudart",
    ],
)

nv_gxf_cc_library(
    name = "clock",
    srcs = ["clock.cpp"],
    hdrs = ["clock.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/utils:time",
    ],
)

nv_gxf_cc_library(
    name = "codelet",
    srcs = ["codelet.cpp"],
    hdrs = ["codelet.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "allocator",
        "receiver",
        "transmitter",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/utils:time",
    ],
)

nv_gxf_cc_library(
    name = "complex",
    hdrs = ["complex.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
        "//third_party:cuda_headers",
    ],
)

nv_gxf_cc_library(
    name = "component_allocator",
    srcs = ["component_allocator.cpp"],
    hdrs = ["component_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "component_factory",
    srcs = ["component_factory.cpp"],
    hdrs = ["component_factory.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "component_allocator",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "connection",
    srcs = ["connection.cpp"],
    hdrs = ["connection.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "receiver",
        "transmitter",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "dlpack_utils",
    srcs = ["dlpack_utils.cpp"],
    hdrs = ["dlpack_utils.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
        "//third_party:cuda_headers",
        "//third_party:cudart",
        "@dlpack",
    ],
)

nv_gxf_cc_library(
    name = "double_buffer_transmitter",
    srcs = ["double_buffer_transmitter.cpp"],
    hdrs = ["double_buffer_transmitter.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "transmitter",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/staging_queue",
    ],
)

nv_gxf_cc_library(
    name = "double_buffer_receiver",
    srcs = ["double_buffer_receiver.cpp"],
    hdrs = ["double_buffer_receiver.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "receiver",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/staging_queue",
    ],
)

nv_gxf_cc_library(
    name = "entity_warden",
    srcs = ["entity_warden.cpp"],
    hdrs = ["entity_warden.hpp"],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "component_factory",
        "entity_resource_helper",
        "//gxf/core:gxf_c_api",
        "//gxf/core:gxf_cpp_api",
        "//gxf/core:entity_item",
    ],
)

nv_gxf_cc_library(
    name = "entity_executor",
    srcs = ["entity_executor.cpp"],
    hdrs = ["entity_executor.hpp"],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "codelet",
        "controller",
        "job_statistics",
        "message_router",
        "network_router",
        "monitor",
        "scheduling_terms",
        "system",
        "//common:nvtx_helper",
        "//gxf/core:gxf_c_api",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "eos",
    hdrs = ["eos.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "extension",
    srcs = ["extension.cpp"],
    hdrs = ["extension.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "component_factory",
        "//gxf/core:expected",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "extension_factory_helper",
    hdrs = ["extension_factory_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
        "//gxf/std:default_extension",
    ],
)

nv_gxf_cc_library(
    name = "extension_loader",
    srcs = ["extension_loader.cpp"],
    hdrs = ["extension_loader.hpp"],
    linkopts = ["-ldl"],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "component_factory",
        "extension",
        "//gxf/core:gxf_c_api",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "default_extension",
    srcs = ["default_extension.cpp"],
    hdrs = ["default_extension.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
        "component_allocator",
        "extension",
        "new_component_allocator",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "file_stream_logger",
    srcs = ["file_stream_logger.cpp"],
    hdrs = ["file_stream_logger.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_c_api",
    ],
)

nv_gxf_cc_library(
    name = "forward",
    hdrs = ["forward.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
        "receiver",
        "transmitter",
    ],
)

nv_gxf_cc_library(
    name = "gather",
    srcs = ["gather.cpp"],
    hdrs = ["gather.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
        "receiver",
        "transmitter",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "greedy_scheduler",
    srcs = ["greedy_scheduler.cpp"],
    hdrs = ["greedy_scheduler.hpp"],
    need_multithread = True,
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "scheduler",
        "resources",
        "//gxf/core:gxf_c_api",
        "//gxf/std/gems/event_list",
    ],
)

nv_gxf_cc_library(
    name = "epoch_scheduler",
    srcs = ["epoch_scheduler.cpp"],
    hdrs = ["epoch_scheduler.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "scheduler",
        "resources",
        "//gxf/core:gxf_c_api",
        "//gxf/std/gems/event_list",
    ],
)

nv_gxf_cc_library(
    name = "metric",
    srcs = ["metric.cpp"],
    hdrs = ["metric.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "memory_buffer",
    hdrs = ["memory_buffer.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:expected",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "monitor",
    srcs = ["monitor.cpp"],
    hdrs = ["monitor.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:component",
    ],
)

nv_gxf_cc_library(
    name = "controller",
    hdrs = ["controller.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:component",
    ],
)

nv_gxf_cc_library(
    name = "multi_thread_scheduler",
    srcs = ["multi_thread_scheduler.cpp"],
    hdrs = ["multi_thread_scheduler.hpp"],
    linkopts = [
        "-pthread",  # For <thread>
    ],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "cpu_thread",
        "entity_resource_helper",
        "resources",
        "scheduler",
        "//gxf/core:gxf_c_api",
        "//gxf/std/gems/event_list",
        "//gxf/std/gems/staging_queue",
        "//gxf/std/gems/timed_job_list",
    ],
)

nv_gxf_cc_library(
    name = "event_based_scheduler",
    srcs = ["event_based_scheduler.cpp"],
    hdrs = ["event_based_scheduler.hpp"],
    linkopts = [
        "-pthread",  # For <thread>
    ],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "cpu_thread",
        "entity_resource_helper",
        "resources",
        "scheduler",
        "//gxf/core:gxf_c_api",
        "//gxf/std/gems/event_list:unique_event_list",
        "//gxf/std/gems/staging_queue",
        "//gxf/std/gems/timed_job_list",
    ],
)

nv_gxf_cc_library(
    name = "resources",
    srcs = ["resources.cpp"],
    hdrs = ["resources.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:component",
    ],
)

nv_gxf_cc_library(
    name = "cpu_thread",
    srcs = ["cpu_thread.cpp"],
    hdrs = ["cpu_thread.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "resources",
        "//gxf/core:component",
    ],
)

nv_gxf_cc_library(
    name = "new_component_allocator",
    hdrs = ["new_component_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "component_allocator",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "program",
    srcs = ["program.cpp"],
    hdrs = ["program.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "connection",
        "entity_executor",
        "entity_warden",
        "ipc_server",
        "job_statistics",
        "message_router",
        "network_router",
        "network_context",
        "monitor",
        "router_group",
        "scheduler",
        "system_group",
        "resources",
        "//common",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "queue",
    srcs = ["queue.cpp"],
    hdrs = ["queue.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "receiver",
    srcs = ["receiver.cpp"],
    hdrs = ["receiver.hpp",
            "transmitter.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "queue",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "router",
    hdrs = ["router.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "connection",
        "receiver",
        "transmitter",
        "network_context",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "router_group",
    srcs = ["router_group.cpp"],
    hdrs = ["router_group.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "router",
        "network_context",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "transmitter",
    srcs = ["transmitter.cpp"],
    hdrs = ["transmitter.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "queue",
        "timestamp",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "message_router",
    srcs = ["message_router.cpp"],
    hdrs = ["message_router.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "connection",
        "network_context",
        "receiver",
        "router",
        "timestamp",
        "topic",
        "transmitter",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "network_router",
    srcs = ["network_router.cpp"],
    hdrs = ["network_router.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "receiver",
        "router",
        "timestamp",
        "transmitter",
        "network_context",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "network_context",
    hdrs = ["network_context.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "typed_message_view",
    hdrs = ["typed_message_view.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "scheduler",
    hdrs = [
        "scheduler.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "entity_executor",
        "router",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "scheduling_terms",
    srcs = [
        "scheduling_condition.cpp",
        "scheduling_terms.cpp",
    ],
    hdrs = [
        "scheduling_condition.hpp",
        "scheduling_term_combiner.hpp",
        "scheduling_term.hpp",
        "scheduling_terms.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "allocator",
        "clock",
        "receiver",
        "timestamp",
        "transmitter",
        "//gxf/core:gxf_c_api",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "system",
    srcs = ["system.cpp"],
    hdrs = ["system.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "system_group",
    srcs = ["system_group.cpp"],
    hdrs = ["system_group.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "system",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "tensor",
    srcs = ["tensor.cpp"],
    hdrs = ["tensor.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "allocator",
        "complex",
        "dlpack_utils",
        "memory_buffer",
        "//gxf/core:gxf_cpp_api",
        "//third_party:cuda_headers",
        "@dlpack",
    ],
)

nv_gxf_cc_library(
    name = "tensor_copier",
    srcs = ["tensor_copier.cpp"],
    hdrs = ["tensor_copier.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
        "tensor",
    ],
)

nv_gxf_cc_library(
    name = "timed_throttler",
    srcs = ["timed_throttler.cpp"],
    hdrs = ["timed_throttler.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "codelet",
        "scheduling_terms",
        "timestamp",
    ],
)

nv_gxf_cc_library(
    name = "timestamp",
    srcs = ["timestamp.cpp"],
    hdrs = ["timestamp.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:expected",
    ],
)

nv_gxf_cc_library(
    name = "unbounded_allocator",
    srcs = ["unbounded_allocator.cpp"],
    hdrs = ["unbounded_allocator.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "allocator",
        "//gxf/core:gxf_cpp_api",
        "//third_party:cudart",
    ],
)

nv_gxf_cc_library(
    name = "vault",
    srcs = ["vault.cpp"],
    hdrs = ["vault.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
    ],
)

nv_gxf_cc_library(
    name = "yaml_file_loader",
    srcs = ["yaml_file_loader.cpp"],
    hdrs = ["yaml_file_loader.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:gxf_cpp_api",
        "@yaml-cpp",
    ],
)

nv_gxf_cc_library(
    name = "job_statistics",
    srcs = ["job_statistics.cpp"],
    hdrs = ["job_statistics.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "ipc_server",
        "//common",
        "//gxf/core:gxf_cpp_api",
        "//gxf/std/gems/utils:exponential_moving_average",
        "//gxf/std/gems/utils:fast_running_median",
        "@nlohmann-json",
    ],
)

nv_gxf_cc_library(
    name = "entity_resource_helper",
    srcs = ["entity_resource_helper.cpp"],
    hdrs = ["entity_resource_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "cpu_thread",
        "resources",
        "//common",
        "//gxf/core:expected",
        "//gxf/core:gxf_c_api",
    ],
)

nv_gxf_cc_library(
    name = "subgraph",
    hdrs = ["subgraph.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "synchronization",
    srcs = ["synchronization.cpp"],
    hdrs = ["synchronization.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "codelet",
        "receiver",
        "timestamp",
        "transmitter",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "synthetic_clock",
    srcs = ["synthetic_clock.cpp"],
    hdrs = ["synthetic_clock.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "clock",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "topic",
    srcs = ["topic.cpp"],
    hdrs = ["topic.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":receiver",
        ":transmitter",
        "//gxf/core:gxf_cpp_api",
    ],
)

nv_gxf_cc_library(
    name = "graph_driver_worker_common",
    srcs = ["graph_driver_worker_common.cpp"],
    hdrs = ["graph_driver_worker_common.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common",
        "//gxf/core:expected",
        "//gxf/core:gxf_cpp_api",
        "@nlohmann-json",
    ],
)

nv_gxf_cc_library(
    name = "graph_driver",
    srcs = ["graph_driver.cpp"],
    hdrs = ["graph_driver.hpp"],
    linkopts = [
        "-pthread",  # For <thread>
    ],
    visibility = ["//visibility:public"],
    deps = [
        "graph_driver_worker_common",
        "ipc_client",
        "ipc_server",
        "system",
        "//gxf/core:gxf_c_api",
        "//gxf/std/gems/queue_thread",
    ],
)

nv_gxf_cc_library(
    name = "graph_worker",
    srcs = ["graph_worker.cpp"],
    hdrs = ["graph_worker.hpp"],
    linkopts = [
        "-pthread",  # For <thread>
    ],
    visibility = ["//visibility:public"],
    deps = [
        "graph_driver_worker_common",
        "ipc_client",
        "ipc_server",
        "system",
        "//gxf/core:gxf_c_api",
        "//gxf/std/gems/queue_thread",
    ],
)

load("//registry/build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

register_extension(
    name = "register_std_ext",
    badges = [""],
    compute_dependencies = ["cuda"],
    extension = "std",
    headers = [
        "BUILD.public",
        "allocator.hpp",
        "async_buffer_receiver.hpp",
        "async_buffer_transmitter.hpp",
        "block_memory_pool.hpp",
        "broadcast.hpp",
        "clock.hpp",
        "codelet.hpp",
        "complex.hpp",
        "component_allocator.hpp",
        "component_factory.hpp",
        "default_extension.hpp",
        "double_buffer_receiver.hpp",
        "double_buffer_transmitter.hpp",
        "eos.hpp",
        "extension.hpp",
        "extension_factory_helper.hpp",
        "forward.hpp",
        "gather.hpp",
        "graph_driver.hpp",
        "graph_driver_worker_common.hpp",
        "graph_worker.hpp",
        "ipc_client.hpp",
        "ipc_server.hpp",
        "memory_buffer.hpp",
        "metric.hpp",
        "monitor.hpp",
        "network_context.hpp",
        "network_router.hpp",
        "new_component_allocator.hpp",
        "queue.hpp",
        "receiver.hpp",
        "resources.hpp",
        "scheduler.hpp",
        "scheduling_condition.hpp",
        "scheduling_term_combiner.hpp",
        "scheduling_term.hpp",
        "scheduling_terms.hpp",
        "system.hpp",
        "tensor.hpp",
        "tensor_copier.hpp",
        "timestamp.hpp",
        "transmitter.hpp",
        "typed_message_view.hpp",
        "unbounded_allocator.hpp",
        "vault.hpp",
    ],
    labels = ["gxf"],
    license_file = ":LICENSE",
    priority = "1",
    python_bindings = [
        "//gxf/std:allocator_pybind.so",
        "//gxf/std:clock_pybind.so",
        "//gxf/std:eos_pybind.so",
        "//gxf/std:receiver_pybind.so",
        "//gxf/std:timestamp_pybind.so",
        "//gxf/std:tensor_pybind.so",
        "//gxf/std:scheduling_terms_pybind.so",
        "//gxf/std:scheduling_condition_pybind.so",
        "//gxf/std:transmitter_pybind.so",
        "//gxf/std:vault_pybind.so",
        "//gxf/core:core_pybind.so",
    ],
    python_sources = [
        "//gxf/std:__init__.py",
        "//gxf/std:Components.py",
        "//gxf/std:standard_graph.py",
        "//gxf/std:standard_methods.py",
        "//gxf/std:compute_entity.py",
        "//gxf/python/gxe:gxe.py",
    ],
    url = "www.nvidia.com",
    uuid = "8ec2d5d6-b5df-48bf-8dee-0252606fdd7e",
    version = "2.6.0",
    visibility = ["//visibility:public"],
)
