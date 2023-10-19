/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/python_codelet/py_codelet.hpp"

PYBIND11_MODULE(pycodelet, m) {
  pybind11::class_<nvidia::gxf::PyCodeletV0>(m, "PyCodeletV0")
      .def("context", [](nvidia::gxf::PyCodeletV0& p) { return p.context(); })
      .def("eid", [](nvidia::gxf::PyCodeletV0& p) { return p.eid(); })
      .def("cid", [](nvidia::gxf::PyCodeletV0& p) { return p.cid(); })
      .def("name", [](nvidia::gxf::PyCodeletV0& p) { return p.name(); })
      .def("get_execution_timestamp", &nvidia::gxf::PyCodeletV0::getExecutionTimestamp)
      .def("get_execution_time", &nvidia::gxf::PyCodeletV0::getExecutionTime)
      .def("get_delta_time", &nvidia::gxf::PyCodeletV0::getDeltaTime)
      .def("get_execution_count", &nvidia::gxf::PyCodeletV0::getExecutionCount)
      .def("is_first_tick", &nvidia::gxf::PyCodeletV0::isFirstTick)
      .def("get_params", &nvidia::gxf::PyCodeletV0::getParams,
           pybind11::return_value_policy::reference);
}
