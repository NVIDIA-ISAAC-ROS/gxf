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

#include "gxf/core/gxf.h"
#include "gxf/core/entity.hpp"
#include "gxf/std/vault.hpp"

namespace {

// Type id for the Vault component.
constexpr gxf_tid_t kGxfStdVaultTid{0x1108cb8d85e44303, 0xba02d27406ee9e65};

}  // namespace

PYBIND11_MODULE(vault_pybind, m) {
  m.doc() = R"pbdoc(
        Python bridge for GXF STD Vault component
        -----------------------

        .. currentmodule:: pygxf

    )pbdoc";
  m.def("store_blocking", [](uint64_t context_idx, int64_t cid, uint64_t count) {
    pybind11::gil_scoped_release release_gil;
    const gxf_context_t context = reinterpret_cast<gxf_context_t>(context_idx);

    // get pointer to queue
    void* pointer;
    gxf_result_t result = GxfComponentPointer(context, cid, kGxfStdVaultTid, &pointer);
    if (result != GXF_SUCCESS) {
      throw pybind11::value_error(GxfResultStr(result));
    }
    nvidia::gxf::Vault* vault = reinterpret_cast<nvidia::gxf::Vault*>(pointer);

    // peek elements from queue
    return vault->storeBlocking(count);
  });
  m.def("store_blocking_for", [](uint64_t context_idx, int64_t cid, uint64_t count,
                                  int64_t duration_ns) {
    pybind11::gil_scoped_release release_gil;
    const gxf_context_t context = reinterpret_cast<gxf_context_t>(context_idx);

    // get pointer to queue
    void* pointer;
    gxf_result_t result = GxfComponentPointer(context, cid, kGxfStdVaultTid, &pointer);
    if (result != GXF_SUCCESS) {
      throw pybind11::value_error(GxfResultStr(result));
    }
    nvidia::gxf::Vault* vault = reinterpret_cast<nvidia::gxf::Vault*>(pointer);

    // peek elements from queue with target duration in nanoseconds
    return vault->storeBlockingFor(count, duration_ns);
  });
  m.def("free", [](uint64_t context_idx, int64_t cid,
                                 const std::vector<gxf_uid_t>& entities) {
    const gxf_context_t context = reinterpret_cast<gxf_context_t>(context_idx);

    // get pointer to queue
    void* pointer;
    gxf_result_t result = GxfComponentPointer(context, cid, kGxfStdVaultTid, &pointer);
    if (result != GXF_SUCCESS) {
      throw pybind11::value_error(GxfResultStr(result));
    }
    nvidia::gxf::Vault* vault = reinterpret_cast<nvidia::gxf::Vault*>(pointer);

    // peek elements from queue
    return vault->free(entities);
  });
}
