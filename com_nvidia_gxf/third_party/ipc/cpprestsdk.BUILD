"""
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

licenses(["notice"])
exports_files(["LICENSES"])

cc_library(
    name = "cpprestsdk",
    includes = [
        "cpprestsdk-2.10.18/Release/include",
        "cpprestsdk-2.10.18/Release/src/pch"
    ],
    hdrs = glob(
        [
            "cpprestsdk-2.10.18/Release/include/cpprest/*.h",
            "cpprestsdk-2.10.18/Release/include/cpprest/details/*.h",
            "cpprestsdk-2.10.18/Release/include/cpprest/details/*.hpp",
            "cpprestsdk-2.10.18/Release/include/cpprest/details/*.dat",
            "cpprestsdk-2.10.18/Release/include/pplx/*.h",
        ],
    ),
    srcs = glob(
        [
            "cpprestsdk-2.10.18/Release/include/cpprest/*.h",
            "cpprestsdk-2.10.18/Release/include/cpprest/*.hpp",
            "cpprestsdk-2.10.18/Release/include/cpprest/include/pplx/*.h",
            "cpprestsdk-2.10.18/Release/include/cpprest/include/pplx/*.hpp",
            "cpprestsdk-2.10.18/Release/include/cpprest/details/*.h",
            "cpprestsdk-2.10.18/Release/include/cpprest/details/*.hpp",
            "cpprestsdk-2.10.18/Release/include/pplx/*.hpp"
        ]
    ) + [
            "cpprestsdk-2.10.18/Release/src/pch/stdafx.h",
            "cpprestsdk-2.10.18/Release/src/http/client/http_client.cpp",
            "cpprestsdk-2.10.18/Release/src/http/client/http_client_impl.h",
            "cpprestsdk-2.10.18/Release/src/http/client/http_client_msg.cpp",
            "cpprestsdk-2.10.18/Release/src/http/client/http_client_asio.cpp",
            "cpprestsdk-2.10.18/Release/src/http/common/x509_cert_utilities.h",
            "cpprestsdk-2.10.18/Release/src/http/common/connection_pool_helpers.h",
            "cpprestsdk-2.10.18/Release/src/http/common/http_compression.cpp",
            "cpprestsdk-2.10.18/Release/src/http/common/http_helpers.cpp",
            "cpprestsdk-2.10.18/Release/src/http/common/http_msg.cpp",
            "cpprestsdk-2.10.18/Release/src/http/common/internal_http_helpers.h",
            "cpprestsdk-2.10.18/Release/src/http/listener/http_listener.cpp",
            "cpprestsdk-2.10.18/Release/src/http/listener/http_listener_msg.cpp",
            "cpprestsdk-2.10.18/Release/src/http/listener/http_server_api.cpp",
            "cpprestsdk-2.10.18/Release/src/http/listener/http_server_impl.h",
            "cpprestsdk-2.10.18/Release/src/http/listener/http_server_asio.cpp",
            "cpprestsdk-2.10.18/Release/src/http/oauth/oauth1.cpp",
            "cpprestsdk-2.10.18/Release/src/http/oauth/oauth2.cpp",
            "cpprestsdk-2.10.18/Release/src/json/json.cpp",
            "cpprestsdk-2.10.18/Release/src/json/json_parsing.cpp",
            "cpprestsdk-2.10.18/Release/src/json/json_serialization.cpp",
            "cpprestsdk-2.10.18/Release/src/uri/uri.cpp",
            "cpprestsdk-2.10.18/Release/src/uri/uri_builder.cpp",
            "cpprestsdk-2.10.18/Release/src/utilities/asyncrt_utils.cpp",
            "cpprestsdk-2.10.18/Release/src/utilities/base64.cpp",
            "cpprestsdk-2.10.18/Release/src/utilities/web_utilities.cpp",
            "cpprestsdk-2.10.18/Release/src/pplx/pplx.cpp",
            "cpprestsdk-2.10.18/Release/src/pplx/pplxlinux.cpp",
            "cpprestsdk-2.10.18/Release/src/pplx/threadpool.cpp",
    ],
    defines = [
        "CPPREST_FORCE_HTTP_CLIENT_ASIO",
        "CPPREST_FORCE_HTTP_LISTENER_ASIO",
        "CPPREST_FORCE_HTTP_CLIENT_ASIO",
        "CPPREST_EXCLUDE_WEBSOCKETS=1",
        "BOOST_BIND_GLOBAL_PLACEHOLDERS"
    ],
    deps = [
        "@boost//:algorithm",
        "@boost//:asio",
        "@boost//:asio_ssl",
        "@boost//:thread",
        "@boringssl//:ssl",
        "@boringssl//:crypto"
    ],
    copts = [
        "-Wall",
        "-Wno-format-truncation",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
