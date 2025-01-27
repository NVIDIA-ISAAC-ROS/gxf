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
    NAME python_codelet
    SOURCES
        py_codelet.cpp
        python_codelet.cpp
    PUBLIC_HEADERS
        pydata.hpp
        py_codelet.hpp
    PUBLIC_DEPENDS
        GXF::core
        GXF::std
        pybind11::pybind11
        Python3::Python
)

#######################################################################################
# Pybind11 extensions
#######################################################################################

pybind11_add_module(pycodelet SHARED "bindings/pycodelet.cpp")
target_link_libraries(pycodelet PUBLIC GXF::core GXF::std GXF::python_codelet)

pybind11_add_module(pydata_pybind SHARED "bindings/pydata.cpp")
target_link_libraries(pydata_pybind PUBLIC GXF::core GXF::std)

install(
    TARGETS pycodelet;pydata_pybind
    EXPORT gxfCoreTargets
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf/multimedia
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/multimedia
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/gxf/multimedia
    COMPONENT Core
)
