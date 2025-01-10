/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <algorithm>
#include "gxf/core/expected.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/std/transmitter.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"

PYBIND11_MODULE(transmitter_pybind, m) {
  pybind11::class_<nvidia::gxf::Transmitter>(m, "Transmitter")
      .def(
          "publish",
          [](nvidia::gxf::Transmitter& t, nvidia::gxf::Entity& other, const int64_t acq_timestamp) {
            auto result = t.publish(other, acq_timestamp);
            return GxfResultStr(nvidia::gxf::ToResultCode(result));
          })
      .def("publish",
           [](nvidia::gxf::Transmitter& t, nvidia::gxf::Entity& other) {
             auto result = t.publish(other, 0);
             return GxfResultStr(nvidia::gxf::ToResultCode(result));
           })
      .def("back_size", &nvidia::gxf::Transmitter::back_size)
      .def("size", &nvidia::gxf::Transmitter::size)
      .def("capacity", &nvidia::gxf::Transmitter::capacity)
      .def("get", [](gxf_context_t context, gxf_uid_t cid, const char* name) {
        auto maybe_transmitter  = nvidia::gxf::CreateHandleFromString<nvidia::gxf::Transmitter>(context, cid, name);
        if (!maybe_transmitter) {
          // GXF_LOG_ERROR("[E%05zu] Couldn't get transmitters", eid);
          throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
        }
        return maybe_transmitter.value().get();
      }, pybind11::return_value_policy::reference);
  pybind11::class_<nvidia::gxf::DoubleBufferTransmitter, nvidia::gxf::Transmitter>(
      m, "DoubleBufferTransmitter").def(pybind11::init<>());
}
