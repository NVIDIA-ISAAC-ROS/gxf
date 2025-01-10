"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

sh_binary(
    name = "run_release_package_validation",
    srcs = ["gxf_package_validator.sh"],
    data = [
        "utility_functions.sh",
    ],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "pre_commit_check",
    srcs = ["pre-commit.sh"],
    data = [
        "before_commit.sh",
        "gxf_package_validator.sh",
        "utility_functions.sh",
    ],
    visibility = ["//visibility:public"],
)
