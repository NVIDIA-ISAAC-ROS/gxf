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
    "nv_gxf_cc_test_group",
    "nv_gxf_app",
    "nv_gxf_test_app",
)

load("//registry/build:registry.bzl", "register_extension")
load("//gxf/python:pygxf.bzl", "nv_pygxf_test")
load("//bzl:cuda.bzl", "cc_cuda_library")

cc_test(
    name = "test_cuda_unit",
    srcs = ["test_cuda_unit.cpp"],
    data = [
        "test_cuda_unit.yaml",
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/core:gxf",
        "//gxf/cuda",
        "//gxf/std",
        "//gxf/cuda:cuda_event",
        "//gxf/cuda:cuda_stream_id",
        "//gxf/cuda:cuda_stream_pool",
        "//common",
        "//third_party:cudart",
        "@yaml-cpp",
        "@gtest//:main",
    ],
)

cc_test(
    name = "cuda_stream_order",
    size = "small",
    srcs = ["test_cuda_stream_order.cpp"],
    data = [
        "//gxf/gxe:manifest",
    ],
    deps = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/test/extensions:test",
        "@gtest//:main",
    ],
)

nv_gxf_cc_test_group(
    name = "tests",
    apps = [
        "gxf/cuda/tests/test_cuda_stream_dotproduct.yaml",
        "gxf/cuda/tests/test_cuda_stream_dotproduct_cuda_allocator.yaml",
        "gxf/cuda/tests/test_cuda_stream_ordered_allocator_cuda_event_st_dotproduct.yaml"
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ],
    manifest_path_hack = "gxf/cuda/tests/",
    deps = ["//gxf/test:test_main"],
)

nv_gxf_test_app(
    name = "test_cuda_allocator",
    srcs = ["test_cuda_allocator.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ]
)

nv_gxf_test_app(
    name = "test_cuda_event_st_dotproduct_mock",
    srcs = [
        "test_cuda_event_st_dotproduct_mock.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_stream_st_dotproduct_mock",
    srcs = [
        "test_cuda_stream_st_dotproduct_mock.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_buffer",
    srcs = [
        "test_cuda_buffer.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_buffer_multi",
    srcs = [
        "test_cuda_buffer_multi.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_buffer_ebs",
    srcs = [
        "test_cuda_buffer_ebs.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_buffer_st",
    srcs = [
        "test_cuda_buffer_st.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_buffer_st_multi",
    srcs = [
        "test_cuda_buffer_st_multi.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_buffer_st_ebs",
    srcs = [
        "test_cuda_buffer_st_ebs.yaml",
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_stream_st_ebs",
    srcs = [
        "test_cuda_stream_st_dotproduct.yaml",
        "test_cuda_st_ebs_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_stream_st_greedy",
    srcs = [
        "test_cuda_stream_st_dotproduct.yaml",
        "test_cuda_st_greedy_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_stream_st_multithread",
    srcs = [
        "test_cuda_stream_st_dotproduct.yaml",
        "test_cuda_st_multithread_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],

)

nv_gxf_test_app(
    name = "test_cuda_stream_st_ping_no_callback_ebs",
    srcs = [
        "test_cuda_stream_st_ping_no_callback.yaml",
        "test_cuda_st_ebs_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_stream_st_ping_no_callback_greedy",
    srcs = [
        "test_cuda_stream_st_ping_no_callback.yaml",
        "test_cuda_st_greedy_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_stream_st_ping_no_callback_multithread",
    srcs = [
        "test_cuda_stream_st_ping_no_callback.yaml",
        "test_cuda_st_multithread_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_event_st_ebs",
    srcs = [
        "test_cuda_event_st_dotproduct.yaml",
        "test_cuda_st_ebs_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_event_st_greedy",
    srcs = [
        "test_cuda_event_st_dotproduct.yaml",
        "test_cuda_st_greedy_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_test_app(
    name = "test_cuda_event_st_multithread",
    srcs = [
        "test_cuda_event_st_dotproduct.yaml",
        "test_cuda_st_multithread_param.yaml"
        ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
)

nv_gxf_cc_test_group(
    name = "tests_cuda_allocator",
    apps = [
        "gxf/cuda/tests/test_cuda_allocator.yaml",
    ],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ],
    manifest_path_hack = "gxf/cuda/tests/",
    deps = ["//gxf/test:test_main"],
)

nv_gxf_cc_extension(
    name = "test_cuda",
    srcs = ["test_cuda_ext.cpp"],
    deps = [
        "test_cuda_helper",
        "//gxf/core:gxf",
        "//gxf/std:tensor",
        "//gxf/std:extension_factory_helper",
    ],
    ext_deps = [
        "//gxf/std",
        "//gxf/cuda",
    ],
)

register_extension(
    name = "register_test_cuda_ext",
    badges = [""],
    extension = "test_cuda",
    uuid = "469d9214-46a5-431a-b7b5-9e598d9fc1db",
    version = "1.6.0",
    headers = [
        "test_cuda_helper.hpp"
    ],
    labels = ["gxf"],
    url = "www.nvidia.com",
    compute_dependencies = ["cuda"],
    local_dependencies = ["//gxf/std:register_std_ext",
                          "//gxf/cuda:register_cuda_ext"],
    priority = "1",
    visibility = ["//visibility:public"],
)

cc_cuda_library(
    name = "convolution",
    srcs = ["convolution.cu.cpp"],
    hdrs = ["convolution.h"],
    visibility = ["//visibility:public"],
)

nv_gxf_cc_library(
    name = "test_cuda_helper",
    hdrs = ["test_cuda_helper.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "convolution",
        "//gxf/cuda:cuda_allocator",
        "//gxf/cuda:cuda_buffer",
        "//gxf/cuda:cuda_event",
        "//gxf/cuda:cuda_stream_id",
        "//gxf/cuda:cuda_stream_pool",
        "//gxf/cuda:cuda_common",
        "//gxf/std",
        "//third_party:cublas",
        "//third_party:cudart",
        "//common",
        "@yaml-cpp"
    ],
)

test_stress_cuda_stream_dotproduct_yaml_files = [
    "test_stress_cuda_stream_dotproduct_gen.yaml",
]

genrule(
    name = "test_stress_cuda_stream_dotproduct_gen",
    outs = test_stress_cuda_stream_dotproduct_yaml_files,
    cmd = "$(location :test_stress_cuda_stream_dotproduct) $(RULEDIR)",
    message = "----- Generating the test_stress_cuda_stream_dotproduct_gen.yaml file",
    tools = [":test_stress_cuda_stream_dotproduct"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_cuda_stream_dotproduct",
    srcs = ["test_stress_cuda_stream_dotproduct.py"],
)

nv_gxf_app(
    name = "test_stress_cuda_stream_dotproduct_app",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_cuda_stream_dotproduct_gen.yaml",
)

test_stress_cuda_stream_dotproduct_cuda_allocator_yaml_files = [
    "test_stress_cuda_stream_dotproduct_cuda_allocator_gen.yaml",
]

genrule(
    name = "test_stress_cuda_stream_dotproduct_cuda_allocator_gen",
    outs = test_stress_cuda_stream_dotproduct_cuda_allocator_yaml_files,
    cmd = "$(location :test_stress_cuda_stream_dotproduct_cuda_allocator) $(RULEDIR)",
    message = "----- Generating the test_cuda_stream_dotproduct_cuda_allocator_gen.yaml file",
    tools = [":test_stress_cuda_stream_dotproduct_cuda_allocator"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "test_stress_cuda_stream_dotproduct_cuda_allocator",
    srcs = ["test_stress_cuda_stream_dotproduct_cuda_allocator.py"],
)

nv_gxf_app(
    name = "test_stress_cuda_stream_dotproduct_cuda_allocator_app",
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
    ],
    gxf_file = "test_stress_cuda_stream_dotproduct_cuda_allocator_gen.yaml",
)

nv_gxf_app(
    name = "test_cuda_allocator_stress_test",
    srcs = ["test_cuda_allocator_stress_test.yaml"],
    extensions = [
        "//gxf/std",
        "//gxf/cuda",
        "//gxf/cuda/tests:test_cuda",
        "//gxf/test/extensions:test",
        "//gxf/sample:sample"
    ]
)
