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

###### Usage:
###### 1. Copy to `<platform>/cmake/GXF` folder in release tarball
###### 2. Point the consuming CMake project to the containing directory with the
######    CMake command line configuration option `-DGXF_DIR:PATH=path/to/gxf-release/<platform>/cmake/GXF`
###### Read more: https://cmake.org/cmake/help/latest/command/find_package.html

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was GXFConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check(GXF_ROOT ${PACKAGE_PREFIX_DIR})
set_and_check(GXF_INCLUDE_DIR ${PACKAGE_PREFIX_DIR})

include(CMakeFindDependencyMacro)

# General dependencies for Holoscan release components.

find_dependency(Threads)
find_dependency(dlpack)
find_dependency(magic_enum)
find_dependency(yaml-cpp)

find_dependency(CUDAToolkit COMPONENTS nvtx3 cudart)
# Workaround for CMake<=3.24: "FindCUDAToolkit" does not initialize CUDA::nvtx3
if(NOT TARGET CUDA::nvtx3)
    find_file(
        CUDA_nvtx3_INCLUDE_DIR
        NAME "nvtx3"
        HINTS /usr/local/cuda-12/include
        REQUIRED
    )
    add_library(CUDA::nvtx3 INTERFACE IMPORTED)
    set_target_properties(CUDA::nvtx3
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CUDA_nvtx3_INCLUDE_DIR}
    )
endif()

if(GXF_FIND_COMPONENTS AND "grpc" IN_LIST GXF_FIND_COMPONENTS)
  find_dependency(Protobuf CONFIG)
  find_dependency(gRPC)
endif()

if(GXF_FIND_COMPONENTS AND "http" IN_LIST GXF_FIND_COMPONENTS)
  find_dependency(cpprestsdk)
endif()

if(GXF_FIND_COMPONENTS AND ("rmm" IN_LIST GXF_FIND_COMPONENTS))
  # Rapids Memory Manager (RMM) is a required dependency for developing with the rmm extension.
  find_dependency(rmm)
endif()

if(GXF_FIND_COMPONENTS AND "stream" IN_LIST GXF_FIND_COMPONENTS)
  find_dependency(nvsci)
endif()

if(GXF_FIND_COMPONENTS AND ("serialization" IN_LIST GXF_FIND_COMPONENTS OR "ucx" IN_LIST GXF_FIND_COMPONENTS))
    find_dependency(Protobuf CONFIG)
endif()

if(GXF_FIND_COMPONENTS AND ("ucx" IN_LIST GXF_FIND_COMPONENTS))
    # TODO: Multiple UCX "find"s will result in target redefinition.
    # We arbitrarily select the `ucx:ucs` target as a proxy workaround
    # to determine whether ucx has already been included.
    if(NOT TARGET ucx::ucs)
        find_dependency(ucx)
    endif()
endif()

if(GXF_FIND_COMPONENTS AND ("python" IN_LIST GXF_FIND_COMPONENTS OR "python_codelet" IN_LIST GXF_FIND_COMPONENTS))
  find_dependency(Python3 COMPONENTS Interpreter Development.Embed)
  find_dependency(pybind11)
endif()

# GXF components available in the binary distribution.
set(GXF_REQUIRED_COMPONENTS
  core
  std
  gxe
)
set(GXF_OPTIONAL_COMPONENTS
    app
    behavior_tree
    cuda
    cuda_test
    grpc
    http
    logger
    multimedia
    network
    npp
    python_codelet
    rmm
    sample
    serialization
    stream
    test_extension
    test_stream_sync_cuda
    ucx
)
set(GXF_COMPONENTS
    ${GXF_REQUIRED_COMPONENTS}
    ${GXF_OPTIONAL_COMPONENTS}
)

include(${CMAKE_CURRENT_LIST_DIR}/GXFTargets.cmake)

foreach(component ${GXF_COMPONENTS})
    if((${component} IN_LIST GXF_REQUIRED_COMPONENTS) OR (${component} IN_LIST GXF_FIND_COMPONENTS))
        # if we have reached this point then dependencies for the component
        # have been found and targets have been imported without errors
        set(GXF_${component}_FOUND TRUE)
    endif()
endforeach()

foreach(component ${GXF_FIND_COMPONENTS})
    if(NOT GXF_${component}_FOUND)
        set(GXF_${component}_FOUND FALSE)
        if(NOT GXF_FIND_QUIETLY)
            message(WARNING "Missing required GXF component \"${component}\"")
        endif()
    endif()
endforeach()

check_required_components(GXF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
