"""
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//gxf/python:pygxf.bzl", "nv_pygxf_test")
load("//gxf:gxf.bzl", "nv_gxf_test_app")

nv_pygxf_test(
    name = "ds_gxf_bridge_test",
    srcs = ["ds_gxf_bridge_test.py"],
    app = "gxf/python/tests/test_core.yaml",
    data = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
        "receiver_subgraph.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/multimedia",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/python_codelet",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/multimedia:multimedia_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/sample:sample_extension",
        "//gxf/std:std_extension",
        "//gxf/std:tensor_pybind",
    ],
)

nv_pygxf_test(
    name = "test_core",
    srcs = ["test_core.py"],
    app = "gxf/python/tests/test_core.yaml",
    data = [
        "receiver_subgraph.yaml",
        "test_ping_cpp_py.yaml",
        "test_ping_py_cpp.yaml",
        "test_ping_py_py.yaml",
        "test_python_app_gxf_graph_wait.yaml",
        "test_python_cuda_stream_dotproduct.yaml",
        "test_python_cuda_stream_dotproduct_cupy.yaml",
        "test_python_cuda_stream_tensor_generator.yaml",
        "test_python_cuda_stream_tensor_generator_cupy.yaml",
        "test_python_data_on_device.yaml",
        "test_python_data_on_device_cupy.yaml",
        "test_python_entity_delete.yaml",
        "test_python_periodic_scheduling_term.yaml",
        "test_python_target_time_scheduling_term.yaml",
        "test_subgraph.yaml",
        "test_py_codelet_invalid.yaml",
        "test_py_codelet_invalid_connection.yaml",
        "test_py_codelet_empty_path.yaml"
    ],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_python_std_tensor",
    srcs = ["test_python_std_tensor.py"],
    app = "gxf/python/tests/test_core.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "pytest",
        "manual",
    ],
    deps = [
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/cuda:cuda_extension",
        "//gxf/std:std_extension",
        "//gxf/sample:sample_extension",
    ],
)

nv_pygxf_test(
    name = "test_python_apis",
    srcs = ["test_python_apis.py"],
    app = "gxf/python/tests/test_core.yaml",
    data = [
        "forward_subgraph.yaml",
        "gather_subgraph.yaml",
        "receiver_subgraph.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/python_codelet",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/sample:sample_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_python_data_transmission",
    srcs = ["test_python_data_transmission.py"],
    app = "gxf/python/tests/test_core.yaml",
    data = [],
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/python_codelet",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/python",
        "//gxf/cuda:cuda_extension",
        "//gxf/std:std_extension",
        "//gxf/sample:sample_extension",
        "//gxf/python_codelet:python_codelet_extension",
    ],
)

nv_pygxf_test(
    name = "test_vector_params",
    srcs = ["test_vector_params.py"],
    app = "gxf/python/tests/test_vector_params.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = ["pytest"],
    deps = [
        "//gxf/python",
    ],
)

nv_pygxf_test(
    name = "test_std_tensor",
    srcs = ["test_std_tensor.py"],
    app = "gxf/python/tests/test_std_tensor.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "exclusive",
        "pytest",
    ],
    deps = [
        "//gxf/python",
        "//gxf/std:scheduling_terms_pybind",
        "//gxf/std:tensor_pybind",
        "//gxf/std:timestamp_pybind",
        "//gxf/std:vault_pybind",
    ],
)

nv_pygxf_test(
    name = "test_py_compute_entity",
    srcs = ["test_py_compute_entity.py"],
    app = "gxf/python/tests/test_core.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
      "pytest",
      "manual",
    ],
    deps = [
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/cuda:cuda_extension",
        "//gxf/std:std_extension",
        "//gxf/sample:sample_extension",
    ],
)

nv_pygxf_test(
    name = "test_std_vault",
    srcs = ["test_std_vault.py"],
    app = "gxf/python/tests/test_std_vault.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/test/extensions:test",
        "//gxf/sample",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "exclusive",
        "manual",
        "pytest",
    ],  # FIXME: flaky on CI/CD
    deps = [
        "//gxf/python",
        "//gxf/std:tensor_pybind",
        "//gxf/std:vault_pybind",
    ],
)

nv_pygxf_test(
    name = "test_boolean_scheduling",
    srcs = ["test_boolean_scheduling.py"],
    app = "gxf/python/tests/test_boolean_scheduling.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_tensor_info",
    srcs = ["test_tensor_info.py"],
    app = "gxf/python/tests/test_tensor_info.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_python_primitive_tensors",
    srcs = ["test_python_primitive_tensors.py"],
    app = "gxf/python/tests/test_python_primitive_tensors.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_tensor_description",
    srcs = ["test_tensor_description.py"],
    app = "gxf/python/tests/test_tensor_description.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = ["pytest"],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_pygxf_test(
    name = "test_count_scheduling",
    srcs = ["test_count_scheduling.py"],
    app = "gxf/python/tests/test_count_scheduling.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/python_codelet",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "exclusive",
        "pytest",
    ],
    deps = [
        "//gxf/cuda:cuda_extension",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
    ],
)

nv_gxf_test_app(
    name = "test_ping_cpp_py",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/sample",
    ],
    srcs = ["//gxf/python/tests:test_ping_cpp_py.yaml"],
    data = [
        "//gxf/python/tests:python_test_codelet",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
        "//gxf/cuda:cuda_extension",
    ],
)

nv_gxf_test_app(
    name = "test_ping_py_py",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/sample",
    ],
    srcs = ["//gxf/python/tests:test_ping_py_py.yaml"],
    data = [
        "//gxf/python/tests:python_test_codelet",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
        "//gxf/cuda:cuda_extension",
    ],
)

nv_gxf_test_app(
    name = "test_ping_py_cpp",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/python_codelet",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/sample",
    ],
    srcs = ["//gxf/python/tests:test_ping_py_cpp.yaml"],
    data = [
        "//gxf/python/tests:python_test_codelet",
        "//gxf/python",
        "//gxf/python_codelet:python_codelet_extension",
        "//gxf/std:std_extension",
        "//gxf/cuda:cuda_extension",
    ]
)

nv_pygxf_test(
    name = "test_python_crash_handling",
    srcs = ["test_python_crash_handling.py"],
    app = "gxf/python/tests/test_core.yaml",
    extensions = [
        "//gxf/std",
        "//gxf/sample",
    ],
    manifest_path_hack = "gxf/python/tests/",
    tags = [
        "manual",
        "pytest",
    ],
    deps = [
        "//gxf/python",
        "//gxf/sample:sample_extension",
        "//gxf/std:std_extension",
    ],
)

filegroup(
    name = "python_test_codelet",
    srcs = glob([
        "*.py",
        "*.yaml",
    ]),
    visibility = ["//visibility:public"],
)

exports_files([
        "test_ping_cpp_py.yaml",
        "test_ping_py_cpp.yaml",
        "test_ping_py_py.yaml",
])
