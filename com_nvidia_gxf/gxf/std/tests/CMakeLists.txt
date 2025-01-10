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

## Test Data

set(GXF_STD_TEST_DATA
    test_epoch_scheduler_app.yaml
    apps/test_ebs_param.yaml
    apps/test_greedy_sched_param.yaml
    apps/test_multi_sched_param.yaml
    apps/test_scheduler_exit.yaml
)

set(GXF_STD_TEST_DATA_OUTPUT "")
foreach(_file ${GXF_STD_TEST_DATA})
    set(_output_file "${CMAKE_CURRENT_BINARY_DIR}/${_file}")
    add_custom_command(
        OUTPUT ${_output_file}
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${_file}" "${_output_file}"
        COMMENT "Copying '${_file}' to ${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND GXF_STD_TEST_DATA_OUTPUT "${_output_file}")
endforeach()
add_custom_target(
       copy_std_test_data
       DEPENDS ${GXF_STD_TEST_DATA_OUTPUT}
)


## Tests

include(GoogleTest)

set(GXF_STD_TEST_SOURCES
    test_boolean_scheduling_term.cpp
    test_eos.cpp
    test_memory_available_scheduling_term.cpp
    test_memory_pool.cpp
    test_multisource_timestamp.cpp
    test_primitive_types.cpp
    test_synchronization.cpp
    test_synthetic_clock.cpp
    test_tensor.cpp
    test_typed_message_view.cpp
    test_unbounded_allocator.cpp
    test_epoch_scheduler.cpp
    test_scheduler.cpp
)

add_executable(GxfStdTest ${GXF_STD_TEST_SOURCES})
target_include_directories(GxfStdTest
    PRIVATE ${CMAKE_SOURCE_DIR}
)
target_link_libraries(GxfStdTest
    PRIVATE
        GTest::gtest_main
        GXF::std
        GXF::core
)
add_dependencies(GxfStdTest
    copy_std_test_data
)
gtest_add_tests(
    TARGET GxfStdTest
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    SOURCES "${GXF_STD_TEST_SOURCES}"
)

