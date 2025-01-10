"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "check_files",
    srcs = ["check_files.py"],
    deps =  [
        "@pip_deps_scripts//pypi__gitpython"
        ],
)

py_binary(
    name = "check_repo",
    srcs = ["check_repo.py"],
    deps =  [
        "@pip_deps_scripts//pypi__gitpython"
        ],
)

py_binary(
    name = "check_commit",
    srcs = ["check_commit.py"],
    deps =  [
        "@pip_deps_scripts//pypi__gitpython"
        ],
)

py_binary(
    name = "core",
    srcs = ["core.py"],
)

py_binary(
    name = "test_core",
    srcs = ["test_core.py"],
)
