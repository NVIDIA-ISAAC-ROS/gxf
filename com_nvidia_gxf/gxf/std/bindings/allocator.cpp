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

#include "gxf/core/component.hpp"
#include "gxf/std/allocator.hpp"

PYBIND11_MODULE(allocator_pybind, m) {
  pybind11::class_<nvidia::gxf::Allocator>(m, "Allocator")
      .def("get_gxf_type", []() { return "nvidia::gxf::Allocator"; })
      .def("free", &nvidia::gxf::Allocator::free)
      .def("allocate", &nvidia::gxf::Allocator::allocate)
      .def("get", [](gxf_context_t context, gxf_uid_t cid, const char* name) {
      auto maybe_allocator = nvidia::gxf::CreateHandleFromString<nvidia::gxf::Allocator>(context, cid, name);
      if (!maybe_allocator) {
        GXF_LOG_ERROR("%s", GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
        throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
      }
      return maybe_allocator.value().get();
    }, pybind11::return_value_policy::reference);
}
