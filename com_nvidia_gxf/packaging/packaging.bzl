"""
 SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("@com_nvidia_gxf//packaging:defs.bzl", "platform_config_path_selector")
load("@rules_pkg//pkg:pkg.bzl", "pkg_tar")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mkdirs", "strip_prefix")

# A simple macro to create a tarball of extension libs, headers and files.
def nv_gxf_pkg_extension(
        name,
        extension,
        build,
        license,
        headers = [],
        binaries = [],
        filegroups = []):

    header_files = name + "_headers"
    pkg_files(
        name = header_files,
        srcs = headers,
        visibility = ["//visibility:public"],
    )

    binary_files = name + "_binaries"
    pkg_files(
        name = binary_files,
        srcs = binaries,
        visibility = ["//visibility:public"],
    )

    build_file = name + "_build"
    pkg_files(
        name = build_file,
        srcs = [build],
        visibility = ["//visibility:public"],
        renames = {
            "BUILD.release" : "BUILD",
        }
    )

    license_file = name + "_license"
    pkg_files(
        name = license_file,
        srcs = [license],
        visibility = ["//visibility:public"],
    )

    # clean up target label
    # Example: //gxf/test/extensions:test to //gxf/test/extensions:libgxf_test.so
    ext_split = extension.split(":")
    if len(ext_split) > 1:
      extension_library = ext_split[0] + ":libgxf_" + ext_split[-1] + ".so"
    else:
      extension_library = "libgxf_" + extension + ".so"

    ext_lib = name + "_lib"
    pkg_files(
        name = ext_lib,
        srcs = [extension_library],
        visibility = ["//visibility:public"],
    )

    # Create extension folder
    pkg_mkdirs(
        name = name + "_dir",
        dirs = [name]
    )

    ext_filegroup = name + "_file_group"
    pkg_filegroup(
        name = ext_filegroup,
        srcs = [
                binary_files,
                header_files,
                build_file,
                license_file,
                ext_lib,
            ],
        prefix = name, # select(platform_config_path_selector),
        visibility = ["//visibility:public"],
    )

    # Perform packaging only if flag is enabled
    ext_pkg = name + "_pkg"
    pkg_tar(
        name = ext_pkg,
        srcs = select({"//conditions:default": [],
                       "@com_nvidia_gxf//packaging:enable_packaging": [ext_filegroup] + filegroups}),
        visibility = ["//visibility:public"],
    )

def nv_gxf_pkg_library(
        name,
        build,
        shared_library = None,
        headers = [],
        binaries = []):

    header_files = name + "_headers"
    pkg_files(
        name = header_files,
        srcs = headers,
        visibility = ["//visibility:public"],
    )

    binary_files = name + "_binaries"
    pkg_files(
        name = binary_files,
        srcs = binaries,
        visibility = ["//visibility:public"],
    )

    build_file = name + "_build"
    pkg_files(
        name = build_file,
        srcs = [build],
        visibility = ["//visibility:public"],
        renames = {
            "BUILD.release" : "BUILD",
        }
    )

    # Perform packaging only if flag is enabled
    cc_pkg = name + "_pkg"
    srcs = [binary_files,
            header_files,
            build_file]
    pkg_tar(
        name = cc_pkg,
        srcs = select({"//conditions:default": [],
                       "@com_nvidia_gxf//packaging:enable_packaging": srcs}),
        visibility = ["//visibility:public"],
    )

def nv_gxf_pkg_filegroup(
        name,
        files = [],
        filegroups = [],
        prefix = None):

    files_pkg = name + "_files"
    pkg_files(
        name = name + "_files",
        srcs = files,
        visibility = ["//visibility:public"],
    )

    filegroup_pkg = name + "_file_group"
    pkg_filegroup(
        name = filegroup_pkg,
        srcs = [files_pkg],
        prefix = prefix, # select(platform_config_path_selector),
        visibility = ["//visibility:public"],
    )

    pkg_name = name + "_pkg"
    pkg_tar(
        name = pkg_name,
        srcs = select({"//conditions:default": [],
                       "@com_nvidia_gxf//packaging:enable_packaging": [filegroup_pkg] + filegroups}),
        visibility = ["//visibility:public"],
    )
