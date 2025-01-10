"""
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "@com_nvidia_gxf//gxf:repo.bzl",
    "nv_gxf_http_archive",
    "nv_gxf_new_git_repository",
    "nv_gxf_git_repository",
    "nv_gxf_new_local_repository",
)

def clean_dep(dep):
    return str(Label(dep))

def ipc_workspace():
    # cpprestsdk for http server implementation
    # Package created from - https://github.com/microsoft/cpprestsdk/archive/refs/tags/2.10.18.tar.gz
    nv_gxf_http_archive(
        name = "cpprestsdk",
        build_file = clean_dep("@com_nvidia_gxf//third_party/ipc:cpprestsdk.BUILD"),
        sha256 = "6bd74a637ff182144b6a4271227ea8b6b3ea92389f88b25b215e6f94fd4d41cb",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/cpprestsdk/2.10.18.tar.gz",
        type = "tar.gz",
        licenses = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/cpprestsdk/license.txt"]
    )

    # Loads boost c++ library (https://www.boost.org/) and
    # custom bazel build support (https://github.com/nelhage/rules_boost/)
    nv_gxf_http_archive(
        name = "com_github_nelhage_rules_boost",
        licenses = ["@com_github_nelhage_rules_boost//:LICENSE"],
        patches = [
            "@com_nvidia_gxf//third_party/ipc:rules_boost.patch",
            "@com_nvidia_gxf//third_party/ipc:boost.patch",
        ],
        sha256 = "dea8de017d8e9709ce55c707196ee1350662199ec94ed0e096d3ddf0339e0f3c",
        type = "tar.gz",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/boost/rules_boost_630cf5dbad418ee8cfa637b1e33125b11807721d.tar.gz",
    )

    # Package created from - https://storage.googleapis.com/grpc-bazel-mirror/github.com/google/boringssl/archive/b9232f9e27e5668bc0414879dcdedb2a59ea75f2.tar.gz
    nv_gxf_http_archive(
        name = "boringssl",
        # Use github mirror instead of https://boringssl.googlesource.com/boringssl
        # to obtain a boringssl archive with consistent sha256
        sha256 = "534fa658bd845fd974b50b10f444d392dfd0d93768c4a51b61263fd37d851c40",
        strip_prefix = "boringssl-b9232f9e27e5668bc0414879dcdedb2a59ea75f2",
        urls = [
            "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/boringssl/b9232f9e27e5668bc0414879dcdedb2a59ea75f2.tar.gz",
        ],
        patches = ["@com_nvidia_gxf//third_party/ipc:boringssl.diff"],
        type = "tar.gz",
        licenses = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/boringssl/LICENSE"]
    )

    # grpc 1.48.0
    # swipat: https://nvbugs/3777206
    # Package created from - https://github.com/grpc/grpc/archive/refs/tags/v1.48.0.tar.gz"
    nv_gxf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "9b1f348b15a7637f5191e4e673194549384f2eccf01fcef7cc1515864d71b424",
        url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/grpc/v1.48.0.tar.gz",
        strip_prefix = "grpc-1.48.0",
        # patch gflags repress warning as error
        patches = ["@com_nvidia_gxf//third_party/ipc:grpc.diff"],
        type = "tar.gz",
        licenses = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/grpc/LICENSE"]
    )

    # Package created from - https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v21.7.tar.gz
    nv_gxf_http_archive(
        name = "com_google_protobuf",
        sha256 = "75be42bd736f4df6d702a0e4e4d30de9ee40eac024c4b845d17ae4cc831fe4ae",
        strip_prefix = "protobuf-21.7",
        urls = [
            "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/protobuf/mirror-v21.7.tar.gz",
        ],
        patches = [
            "@com_nvidia_gxf//third_party:protobuf.patch",
        ],
        patch_args = ["-p1"],
        type = "tar.gz",
        licenses = ["TBD"],
    )
