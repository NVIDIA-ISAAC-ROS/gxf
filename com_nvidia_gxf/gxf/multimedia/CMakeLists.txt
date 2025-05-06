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

set(GXF_MULTIMEDIA_SOURCES
        audio.cpp
        multimedia.cpp
        video.cpp)
set(GXF_MULTIMEDIA_HEADERS
        audio.hpp
        camera.hpp
        video.hpp)
add_library(multimedia SHARED ${GXF_MULTIMEDIA_SOURCES})
set_target_properties(multimedia
  PROPERTIES
    OUTPUT_NAME "gxf_multimedia"
    PUBLIC_HEADER "${GXF_MULTIMEDIA_HEADERS}"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../core:$ORIGIN/../std"
)

target_include_directories(multimedia PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                                       $<INSTALL_INTERFACE:include>)
target_link_libraries(multimedia PUBLIC GXF::core GXF::std)
add_library(GXF::multimedia ALIAS multimedia)

include(GNUInstallDirs)
install(
  TARGETS multimedia
  EXPORT gxfCoreTargets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/multimedia
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/multimedia
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/multimedia
  COMPONENT multimedia)
set(GXF_EXTENSION_LIBRARY_TARGETS "multimedia;${GXF_EXTENSION_LIBRARY_TARGETS}" PARENT_SCOPE)

#######################################################################################
# Pybind11 extensions
#######################################################################################

pybind11_add_module(video_pybind SHARED WITHOUT_SOABI "bindings/video.cpp")
set_target_properties(video_pybind PROPERTIES OUTPUT_NAME "video_pybind" PREFIX "")
target_link_libraries(video_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(camera_pybind SHARED WITHOUT_SOABI "bindings/camera.cpp")
set_target_properties(camera_pybind PROPERTIES OUTPUT_NAME "camera_pybind" PREFIX "")
target_link_libraries(camera_pybind PUBLIC GXF::core GXF::std)

install(
    TARGETS video_pybind;camera_pybind
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/multimedia
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/multimedia
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/multimedia
    COMPONENT multimedia
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
