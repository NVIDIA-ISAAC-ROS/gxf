/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/std/clock.hpp"
#include "gxf/core/component.hpp"

PYBIND11_MODULE(clock_pybind, m) {
  pybind11::class_<nvidia::gxf::Clock>(m, "Clock")
      .def("timestamp", &nvidia::gxf::Clock::timestamp)
      .def("time", &nvidia::gxf::Clock::time)
      .def("get", [](gxf_context_t context, gxf_uid_t cid, const char* name) {
          auto handle = nvidia::gxf::CreateHandleFromString<nvidia::gxf::Clock>(context, cid, name);
          if (!handle) {
            // GXF_LOG_ERROR("[E%05zu] Couldn't get a handle to clock: %s", this->eid(), name.c_str());
            throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
          }
          return handle.value().get();
      }, pybind11::return_value_policy::reference);
  pybind11::class_<nvidia::gxf::RealtimeClock, nvidia::gxf::Clock>(m, "RealtimeClock").def(pybind11::init<>());
  pybind11::class_<nvidia::gxf::ManualClock, nvidia::gxf::Clock>(m, "ManualClock").def(pybind11::init<>());
}
