/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/python_codelet/pydata.hpp"
#include <pybind11/pybind11.h>
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"

PYBIND11_MODULE(pydata_pybind, m) {
  m.doc() = R"pbdoc(
        Python bridge for accessing pydata of messages
        -----------------------

        .. currentmodule:: pygxf

    )pbdoc";

  pybind11::class_<nvidia::gxf::PyData>(m, "PyData")
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, pybind11::object& data, const char* name) {
            auto result = e.add<nvidia::gxf::PyData>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            auto pydata = result.value().get();
            auto result_1 = pydata->setData(data);
            if (!result_1) { throw pybind11::value_error(GxfResultStr(result_1.error())); }
            return;
          },
          pybind11::arg("entity"), pybind11::arg("data"), pybind11::arg("name") = nullptr)
      .def_static(
          "get_from_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto maybe_pydata = e.get<nvidia::gxf::PyData>(name);
            if (!maybe_pydata) {
              GXF_LOG_ERROR("Error getting pydata called %s", name);
              throw pybind11::value_error("error getting pydata");
            }
            auto maybe_pyobj = maybe_pydata.value().get()->getData();
            if (!maybe_pyobj) { throw pybind11::value_error(maybe_pydata.get_error_message()); }
            return maybe_pyobj.value();
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference);
}
