/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/core/gxf.h"
#include "gxf/std/eos.hpp"

PYBIND11_MODULE(eos_pybind, m) {
  pybind11::class_<nvidia::gxf::EndOfStream>(m, "EndOfStream")
      .def_static(
          "get_from_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto maybe_eos = e.get<nvidia::gxf::EndOfStream>(name);
            if (!maybe_eos) {
              GXF_LOG_ERROR("Error getting eos called %s", name);
              throw pybind11::value_error("error getting tensor");
            }
            auto eos = maybe_eos.value().get();
            return eos;
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& message, int64_t stream_id = -1, const char* name = nullptr) {
            auto eos = message.add<nvidia::gxf::EndOfStream>(name);
            if (!eos) {
              GXF_LOG_ERROR("Error adding EOS to entity");
              throw pybind11::value_error(GxfResultStr(eos.error()));
            }
            eos.value()->stream_id(stream_id);
            return;
          },
          pybind11::arg("entity"), pybind11::arg("stream_id"), pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& message, nvidia::gxf::EndOfStream& eos, const char* name = nullptr) {
            auto eos_ = message.add<nvidia::gxf::EndOfStream>(name);
            if (!eos_) {
              GXF_LOG_ERROR("Error adding EOS to entity");
              throw pybind11::value_error(GxfResultStr(eos_.error()));
            }
            eos_.value()->stream_id(eos.stream_id());
            return;
          },
          pybind11::arg("entity"), pybind11::arg("eos"), pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def(
          "stream_id",
          [](nvidia::gxf::EndOfStream& eos) {
            int64_t stream_id = eos.stream_id();
            if (stream_id < 0) {
              GXF_LOG_INFO("Negative stream id[%ld]: EOS for the entire pipeline", stream_id);
            }
            return stream_id;
          });
}
