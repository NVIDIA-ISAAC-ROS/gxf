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

gxf_core_add_extension_library(
    NAME multimedia
    SOURCES
        audio.cpp
        multimedia.cpp
        video.cpp
    PUBLIC_HEADERS
        audio.hpp
        camera.hpp
        video.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::std
)


#######################################################################################
# Pybind11 extensions
#######################################################################################

pybind11_add_module(video_pybind SHARED "bindings/video.cpp")
target_link_libraries(video_pybind PUBLIC GXF::core GXF::std)

pybind11_add_module(camera_pybind SHARED "bindings/camera.cpp")
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
    COMPONENT Core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
