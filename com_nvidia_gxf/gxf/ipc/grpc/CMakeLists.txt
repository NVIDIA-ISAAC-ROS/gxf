# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

###################################################################################################
# Workaround: Fetch "health.v1" proto file from gRPC repository.
# The gRPC v1.48.0 CMake package excludes the "health.v1" module,
# though it is available in Bazel.
# See: https://github.com/grpc/grpc-proto/tree/master?tab=readme-ov-file#grpc-proto
# "This repository contains the canonical versions of common protocol definitions for
# peripheral services around gRPC such as health checking and load balancing.
# Non-Bazel users are expected to copy the proto files from this repo as needed."
#
# Since `grpc-proto` is not tagged and `health.proto` is available in the tagged gRPC source repo,
# we fetch from the gRPC tagged source.
###################################################################################################

find_package(gRPC CONFIG REQUIRED)
include(FetchContent)
set(grpc_health_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/grpc_health_proto-src")
set(grpc_health_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/grpc_health_proto-build")
FetchContent_Declare(
    grpc_health_proto
    URM_HASH "MD5=93c62f9d6b1925cf3e8822d590484589"
    DOWNLOAD_NO_EXTRACT TRUE
    SOURCE_DIR ${grpc_health_SOURCE_DIR}
    URL "https://raw.githubusercontent.com/grpc/grpc/v${gRPC_VERSION}/src/proto/grpc/health/v1/health.proto"
)
FetchContent_MakeAvailable(grpc_health_proto)

set(grpc_health_proto_SOURCES ${grpc_health_SOURCE_DIR}/health.proto)
set(grpc_health_GEN_SOURCES
    ${grpc_health_BUILD_DIR}/health.pb.cc
    ${grpc_health_BUILD_DIR}/health.grpc.pb.cc)
set(grpc_health_GEN_HEADERS
    ${grpc_health_BUILD_DIR}/health.pb.h
    ${grpc_health_BUILD_DIR}/health.grpc.pb.h
)
add_custom_command(
    OUTPUT ${grpc_health_GEN_HEADERS} ${grpc_health_GEN_SOURCES}
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${grpc_health_BUILD_DIR}"
      --cpp_out "${grpc_health_BUILD_DIR}"
      -I "${grpc_health_SOURCE_DIR}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${grpc_health_proto_SOURCES}"
    DEPENDS "${grpc_health_proto_SOURCES}"
)

###################################################################################################
# GXF gRPC extension
###################################################################################################

set(grpc_ext_SOURCES
    grpc_client.cpp
    grpc_ext.cpp
    grpc_server.cpp
)
set(grpc_ext_HEADERS
    grpc_client.hpp
    grpc_server.hpp
)

set(grpc_ext_proto_SOURCES grpc_service.proto)
set(grpc_ext_GEN_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/grpc_service.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/grpc_service.grpc.pb.cc)
set(grpc_ext_GEN_HEADERS
    ${CMAKE_CURRENT_BINARY_DIR}/grpc_service.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/grpc_service.grpc.pb.h
)

add_custom_command(
    OUTPUT ${grpc_ext_GEN_HEADERS} ${grpc_ext_GEN_SOURCES}
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
      -I "${CMAKE_CURRENT_SOURCE_DIR}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${grpc_ext_proto_SOURCES}"
    DEPENDS "${grpc_ext_proto_SOURCES}"
)
add_custom_target(generate_grpc_ext_sources ALL
    DEPENDS
        ${grpc_ext_GEN_HEADERS}
        ${grpc_ext_GEN_SOURCES}
    COMMENT "Generate service descriptors with gRPC"
)

add_library(grpc_ext SHARED
    ${grpc_ext_SOURCES}
    ${grpc_ext_GEN_SOURCES}
    ${grpc_ext_GEN_HEADERS}
    ${grpc_health_GEN_HEADERS}
    ${grpc_health_GEN_SOURCES}
)
set_target_properties(grpc_ext
    PROPERTIES
    PUBLIC_HEADER "${grpc_ext_HEADERS}"
    OUTPUT_NAME "gxf_grpc"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../core:$ORIGIN/../std"
)
# Reflect the grpc_health module workaround above for non-Bazel build systems
target_compile_definitions(grpc_ext PRIVATE CMAKE_BUILD)
target_include_directories(grpc_ext
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${grpc_health_BUILD_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(grpc_ext
    PUBLIC
        GXF::core
        GXF::std
    PRIVATE
        gRPC::grpc++
        gRPC::grpc++_reflection
)
add_dependencies(grpc_ext
    generate_grpc_ext_sources
)
add_library(GXF::grpc_ext ALIAS grpc_ext)
set(GXF_EXTENSION_LIBRARY_TARGETS "${GXF_EXTENSION_LIBRARY_TARGETS};grpc_ext" PARENT_SCOPE)

install(
    TARGETS grpc_ext
    EXPORT GXFExtensionTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/ipc/grpc
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/ipc/grpc
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/ipc/grpc
    COMPONENT gRPC
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
