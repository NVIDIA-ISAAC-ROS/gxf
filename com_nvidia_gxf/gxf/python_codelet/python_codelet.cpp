/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/python_codelet/py_codelet.hpp"
#include "gxf/python_codelet/pydata.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0x787daddc1c3411ec, 0x96210242ac130002, "PythonCodeletExtension",
                          "An Extension for implementing Python Codelet", "NVIDIA", "0.6.0",
                          "NVIDIA");
  GXF_EXT_FACTORY_SET_DISPLAY_INFO("Python Codelet Extension", "Python",
                                   "GXF Python Codelet Extension");
  GXF_EXT_FACTORY_ADD(
      0xcd8b08c2f643483f, 0xf33b02bfa75c23fb, nvidia::gxf::PyCodeletV0, nvidia::gxf::Codelet,
      "A wrapper codelet for implementing python codelets which interfaces with CodeletAdapter");
  GXF_EXT_FACTORY_ADD_0(0x890b6e9c5bd311ee, 0x8c990242ac120002, nvidia::gxf::PyData,
                      "A component to wrap python object to send over transmitter");
GXF_EXT_FACTORY_END()
