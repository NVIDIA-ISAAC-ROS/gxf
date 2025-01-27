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

add_subdirectory(gems)

#######################################################################################
# Std (Internal)
#
# These sources are depended on by targets in both the `core` and `std` directories.
# This "internal" target is required to manage the circular dependency among `std`
# and `core` directories.
#######################################################################################
set(GXF_STD_INTERNAL_SOURCES
    allocator.cpp
    async_buffer_receiver.cpp
    async_buffer_transmitter.cpp
    block_memory_pool.cpp
    broadcast.cpp
    clock.cpp
    codelet.cpp
    component_allocator.cpp
    component_factory.cpp
    connection.cpp
    cpu_thread.cpp
    default_extension.cpp
    dlpack_utils.cpp
    double_buffer_receiver.cpp
    double_buffer_transmitter.cpp
    entity_executor.cpp
    entity_resource_helper.cpp
    entity_warden.cpp
    epoch_scheduler.cpp
    event_based_scheduler.cpp
    extension.cpp
    extension_loader.cpp
    file_stream_logger.cpp
    gather.cpp
    graph_driver.cpp
    graph_driver_worker_common.cpp
    graph_worker.cpp
    greedy_scheduler.cpp
    job_statistics.cpp
    message_router.cpp
    metric.cpp
    monitor.cpp
    multi_thread_scheduler.cpp
    network_router.cpp
    program.cpp
    queue.cpp
    receiver.cpp
    resources.cpp
    router_group.cpp
    scheduling_condition.cpp
    scheduling_terms.cpp
    synchronization.cpp
    synthetic_clock.cpp
    system.cpp
    system_group.cpp
    tensor_copier.cpp
    tensor.cpp
    timed_throttler.cpp
    timestamp.cpp
    topic.cpp
    transmitter.cpp
    unbounded_allocator.cpp
    vault.cpp
    yaml_file_loader.cpp
)

set(GXF_STD_INTERNAL_PUBLIC_HEADERS
    allocator.hpp
    async_buffer_receiver.hpp
    async_buffer_transmitter.hpp
    block_memory_pool.hpp
    broadcast.hpp
    clock.hpp
    codelet.hpp
    complex.hpp
    component_allocator.hpp
    component_factory.hpp
    connection.hpp
    controller.hpp
    cpu_thread.hpp
    default_extension.hpp
    dlpack_utils.hpp
    double_buffer_receiver.hpp
    double_buffer_transmitter.hpp
    entity_executor.hpp
    entity_resource_helper.hpp
    entity_warden.hpp
    eos.hpp
    event_based_scheduler.hpp
    extension_factory_helper.hpp
    extension.hpp
    extension_loader.hpp
    file_stream_logger.hpp
    forward.hpp
    gather.hpp
    graph_driver.hpp
    graph_driver_worker_common.hpp
    graph_worker.hpp
    ipc_server.hpp
    job_statistics.hpp
    memory_buffer.hpp
    message_router.hpp
    metric.hpp
    monitor.hpp
    multi_thread_scheduler.hpp
    network_context.hpp
    network_router.hpp
    new_component_allocator.hpp
    program.hpp
    queue.hpp
    receiver.hpp
    resources.hpp
    router_group.hpp
    router.hpp
    scheduler.hpp
    scheduling_condition.hpp
    subgraph.hpp
    synchronization.hpp
    synthetic_clock.hpp
    system_group.hpp
    system.hpp
    tensor_copier.hpp
    tensor.hpp
    timed_throttler.hpp
    timestamp.hpp
    topic.hpp
    transmitter.hpp
    typed_message_view.hpp
    unbounded_allocator.hpp
    vault.hpp
    yaml_file_loader.hpp
)

add_library(std_internal OBJECT
    ${GXF_STD_INTERNAL_SOURCES}
    $<TARGET_OBJECTS:std_gems_utils_internal>
)
set_target_properties(std_internal
    PROPERTIES PUBLIC_HEADER "${GXF_STD_INTERNAL_PUBLIC_HEADERS}"
)

# nlohmann_json header-only library is a private, compile-only dependency for std_internal.
# std_internal is an OBJECT library for purposes of handling circular dependencies, thus
# adding the nlohmann_json target to PRIVATE linked libraries would propagate in
# GXF export rules. To avoid an unnecessary public dependency on nlohmann_json,
# we reference include directories at compilation and avoid depending on its targets.
get_target_property(nlohmann_json_interface_include_dirs
    nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES
)
set(nlohmann_json_build_include_dirs "")
foreach(_include_dir ${nlohmann_json_interface_include_dirs})
    list(APPEND nlohmann_json_build_include_dirs "$<BUILD_INTERFACE:${_include_dir}>")
endforeach()
target_include_directories(std_internal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${nlohmann_json_build_include_dirs}
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(std_internal
    PUBLIC
        GXF::common
        GXF::core_internal
        GXF::std::gems::event_list
        GXF::std::gems::staging_queue
        GXF::std::gems::timed_job_list
        CUDA::cudart
        CUDA::nvtx3
        dlpack::dlpack
        Threads::Threads
        yaml-cpp::yaml-cpp
    PRIVATE
        GXF::std::gems::utils_internal
)
add_library(GXF::std_internal ALIAS std_internal)

#######################################################################################
# Standard extension
#
# `libgxf_std.so` depends on `libgxf_core.so` and is dynamically loaded at runtime.
#######################################################################################
set(GXF_STD_SOURCES std.cpp)
set(GXF_STD_HEADERS
    epoch_scheduler.hpp
    greedy_scheduler.hpp
    scheduling_term.hpp
    scheduling_terms.hpp
    scheduling_term_combiner.hpp
)

add_library(std SHARED ${GXF_STD_SOURCES})
set_target_properties(std
    PROPERTIES
    PUBLIC_HEADER "${GXF_STD_HEADERS}"
    OUTPUT_NAME "gxf_std"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../core"
)
target_include_directories(std
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(std
    PUBLIC
        GXF::common
        GXF::core
        GXF::std_internal
)
add_library(GXF::std ALIAS std)
set(GXF_EXTENSION_LIBRARY_TARGETS "std;${GXF_EXTENSION_LIBRARY_TARGETS}" PARENT_SCOPE)

#######################################################################################
# Pybind11 extensions
#######################################################################################

pybind11_add_module(vault_pybind SHARED "bindings/vault.cpp")
target_link_libraries(vault_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(allocator_pybind SHARED "bindings/allocator.cpp")
target_link_libraries(allocator_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(clock_pybind SHARED "bindings/clock.cpp")
target_link_libraries(clock_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(receiver_pybind SHARED "bindings/receiver.cpp")
target_link_libraries(receiver_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(tensor_pybind SHARED "bindings/tensor.cpp")
target_link_libraries(tensor_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(scheduling_terms_pybind SHARED "bindings/scheduling_terms.cpp")
target_link_libraries(scheduling_terms_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(timestamp_pybind SHARED "bindings/timestamp.cpp")
target_link_libraries(timestamp_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(eos_pybind SHARED "bindings/eos.cpp")
target_link_libraries(eos_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(transmitter_pybind SHARED "bindings/transmitter.cpp")
target_link_libraries(transmitter_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(scheduling_condition_pybind SHARED "bindings/scheduling_condition.cpp")
target_link_libraries(scheduling_condition_pybind PUBLIC GXF::core GXF::std)

install(
    TARGETS std_internal;std;vault_pybind;allocator_pybind;clock_pybind;receiver_pybind;tensor_pybind;scheduling_terms_pybind;timestamp_pybind;eos_pybind;transmitter_pybind;scheduling_condition_pybind
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/std
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/std
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/std
    COMPONENT Core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
