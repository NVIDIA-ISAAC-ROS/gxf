/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/pybind11.h>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/timestamp.hpp"

PYBIND11_MODULE(timestamp_pybind, m) {
  m.doc() = R"pbdoc(
        Python bridge for accessing timestamps of messages
        -----------------------

        .. currentmodule:: pygxf

    )pbdoc";

  // Exposes the acquisition time of struct
  m.def("as_timestamp", [](uint64_t context_idx, gxf_uid_t eid, const char* timestamp_field) {
    const gxf_context_t context = reinterpret_cast<gxf_context_t>(context_idx);
    auto entity = nvidia::gxf::Entity::Shared(context, eid);
    auto maybe_timestamp = entity.value().get<nvidia::gxf::Timestamp>(timestamp_field);
    if (!maybe_timestamp) {
      throw pybind11::value_error("Field with matching name does not exist");
    }
    return pybind11::make_tuple(maybe_timestamp.value()->acqtime, maybe_timestamp.value()->pubtime);
  });
  pybind11::class_<nvidia::gxf::Timestamp>(m, "Timestamp")
    .def_static(
      "get_from_entity",
      [](nvidia::gxf::Entity& e, const char* name = nullptr) {
        auto maybe_timestamp = e.get<nvidia::gxf::Timestamp>(name);
        if (!maybe_timestamp) {
          throw pybind11::value_error("Field with matching name does not exist");
        }
        return pybind11::make_tuple(maybe_timestamp.value()->acqtime, maybe_timestamp.value()->pubtime);
      },
      pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
      pybind11::return_value_policy::reference)
    .def_static(
      "add_to_entity",
      [](nvidia::gxf::Entity& e, int64_t acqtime, int64_t pubtime, const char* name = nullptr) {
        auto result = e.add<nvidia::gxf::Timestamp>(name);
        if (!result) {
          throw pybind11::value_error(GxfResultStr(result.error()));
        }
        nvidia::gxf::Timestamp& timestamp = *result.value();
        timestamp.acqtime = acqtime;
        timestamp.pubtime = pubtime;
        return result.value().get();
      },
      pybind11::arg("entity") = nullptr, pybind11::arg("acqtime"), pybind11::arg("pubtime"),
        pybind11::arg("name") = nullptr,
      pybind11::return_value_policy::reference);
}
