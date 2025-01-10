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

###### WARNING: This file has been adapted from CMake generated output.       ######
###### It is intended as a temporary workaround and will be removed in favor  ######
###### of first-class CMake support in the future.                            ######

#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

get_filename_component(GXF_PLATFORM_LIBRARY_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(GXF_PLATFORM_LIBRARY_PREFIX "${GXF_PLATFORM_LIBRARY_PREFIX}" PATH)
get_filename_component(GXF_PLATFORM_LIBRARY_PREFIX "${GXF_PLATFORM_LIBRARY_PREFIX}" PATH)
get_filename_component(GXF_PLATFORM_LIBRARY_PREFIX "${GXF_PLATFORM_LIBRARY_PREFIX}" NAME)

set(GXF_IMPORTED_TARGETS "")

#----------------------------------------------------------------
# Required components: common, logger, core, std, gxe

set_target_properties(GXF::logger PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/logger/libgxf_logger.so"
  IMPORTED_SONAME_RELEASE "libgxf_logger.so"
)
list(APPEND GXF_IMPORTED_TARGETS "logger")

# Import target "GXF::core" for configuration "Release"
set_target_properties(GXF::core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/core/libgxf_core.so"
  IMPORTED_SONAME_RELEASE "libgxf_core.so"
  )
list(APPEND GXF_IMPORTED_TARGETS "core")

# Import target "GXF::core_static" for configuration "Release"
set_target_properties(GXF::core_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/core/libgxf.lo"
  )
list(APPEND GXF_IMPORTED_TARGETS "core_static")

# Import target "GXF::std" for configuration "Release"
set_target_properties(GXF::std PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/std/libgxf_std.so"
  IMPORTED_SONAME_RELEASE "libgxf_std.so"
  )
list(APPEND GXF_IMPORTED_TARGETS "std")

# Import target "GXF::std_static" for configuration "Release"
set_property(TARGET GXF::std_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GXF::std_static PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/std/libgxf_std_static.lo"
  IMPORTED_SONAME_RELEASE "libgxf_std_static.lo"
  )
list(APPEND GXF_IMPORTED_TARGETS "std_static")

# Import target "GXF::gxe" for configuration "Release"
set_property(TARGET GXF::gxe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GXF::gxe PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/gxe/gxe"
  )
list(APPEND GXF_IMPORTED_TARGETS "gxe")

#----------------------------------------------------------------
# Optional components

if(TARGET GXF::behavior_tree)
  set_target_properties(GXF::behavior_tree PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/behavior_tree/libgxf_behavior_tree.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS behavior_tree)
endif()

if(TARGET GXF::cuda)
  set_target_properties(GXF::cuda PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/cuda/libgxf_cuda.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS cuda)
endif()

if(TARGET GXF::cuda_test)
  set_target_properties(GXF::cuda_test PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/cuda/tests/libgxf_test_cuda.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS cuda_test)
endif()

if(TARGET GXF::app)
  set_target_properties(GXF::app PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/app/libgxf_app.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS app)
endif()

if(TARGET GXF::grpc)
  set_target_properties(GXF::grpc PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/ipc/grpc/libgxf_grpc.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS grpc)
endif()

if(TARGET GXF::http)
  set_target_properties(GXF::http PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/ipc/http/libgxf_http.so"
  )
  list(APPEND GXF_IMPORTED_TARGETS http)
endif()

if(TARGET GXF::multimedia)
  set_target_properties(GXF::multimedia PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/multimedia/libgxf_multimedia.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS multimedia)
endif()

if(TARGET GXF::network)
  set_target_properties(GXF::network PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/network/libgxf_network.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS network)
endif()

if(TARGET GXF::npp)
  set_target_properties(GXF::npp PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/npp/libgxf_npp.so"
  )
  list(APPEND GXF_IMPORTED_TARGETS npp)
endif()

if(TARGET GXF::python_codelet)
  set_target_properties(GXF::python_codelet PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/python_codelet/libgxf_python_codelet.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS python_codelet)
endif()

if(TARGET GXF::rmm)
  set_target_properties(GXF::rmm PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/rmm/libgxf_rmm.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS rmm)
endif()

if(TARGET GXF::sample)
  set_target_properties(GXF::sample PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/sample/libgxf_sample.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS sample)
endif()

if(TARGET GXF::serialization)
  set_target_properties(GXF::serialization PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/serialization/libgxf_serialization.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS serialization)
endif()

if(TARGET GXF::stream)
  set_target_properties(GXF::stream PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/stream/libgxf_stream.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS stream)
endif()

if(TARGET GXF::test_stream_sync_cuda)
  set_target_properties(GXF::test_stream_sync_cuda PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/stream/tests/libgxf_test_stream_sync_cuda.so"
  )
  list(APPEND GXF_IMPORTED_TARGETS test_stream_sync_cuda)
endif()

if(TARGET GXF::test_extension)
  set_target_properties(GXF::test_extension PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/test/extensions/libgxf_test.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS test_extension)
endif()

if(TARGET GXF::ucx)
  set_target_properties(GXF::ucx PROPERTIES
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/${GXF_PLATFORM_LIBRARY_PREFIX}/ucx/libgxf_ucx.so"
    )
  list(APPEND GXF_IMPORTED_TARGETS ucx)
endif()

foreach(_target_suffix ${GXF_IMPORTED_TARGETS})
  set(_target "GXF::${_target_suffix}")
  set_property(TARGET ${_target} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
  get_target_property(_release_location ${_target} IMPORTED_LOCATION_RELEASE)
  set_property(TARGET ${_target} PROPERTY IMPORTED_LOCATION ${_release_location})
  list(APPEND _cmake_import_check_targets "${_target}")
  list(APPEND _cmake_import_check_files_for_${_target} ${_release_location})
endforeach()

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
