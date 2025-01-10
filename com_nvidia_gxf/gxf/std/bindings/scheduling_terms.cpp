/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/core/component.hpp"
#include "gxf/std/scheduling_terms.hpp"

PYBIND11_MODULE(scheduling_terms_pybind, m) {
  pybind11::class_<nvidia::gxf::SchedulingTerm>(m, "SchedulingTerm");
  pybind11::class_<nvidia::gxf::BooleanSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "BooleanSchedulingTerm")
      .def(pybind11::init<>())
      .def("make_enable_tick", [](nvidia::gxf::BooleanSchedulingTerm& bst) { bst.enable_tick(); })

      .def("make_disable_tick", [](nvidia::gxf::BooleanSchedulingTerm& bst) { bst.disable_tick(); })

      .def("check_tick_enable",
           [](nvidia::gxf::BooleanSchedulingTerm& bst) { return bst.checkTickEnabled(); })
      .def(
          "get",
          [](gxf_context_t context, gxf_uid_t cid, const char* name) {
            auto handle = nvidia::gxf::CreateHandleFromString<nvidia::gxf::BooleanSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to BooleanSchedulingTerm: %s", cid,
                            name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::CountSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "CountSchedulingTerm")
      .def(pybind11::init<>())
      .def("check",
           [](nvidia::gxf::CountSchedulingTerm& cst) {
             nvidia::gxf::SchedulingConditionType type;
             int64_t target_timestamp = 0;
             cst.check_abi(0, &type, &target_timestamp);
             return type;
           })
      .def(
          "get",
          [](gxf_context_t context, gxf_uid_t cid, const char* name) {
            auto handle = nvidia::gxf::CreateHandleFromString<nvidia::gxf::CountSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to CountSchedulingTerm: %s", cid, name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::PeriodicSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "PeriodicSchedulingTerm")
      .def(pybind11::init<>())
      .def("check",
           [](nvidia::gxf::PeriodicSchedulingTerm& pst, int64_t timestamp) {
             nvidia::gxf::SchedulingConditionType type;
             int64_t target_timestamp;
             pst.check_abi(timestamp, &type, &target_timestamp);
             return type;
           })
      .def("recess_period_ns",
           [](nvidia::gxf::PeriodicSchedulingTerm& pst) { return pst.recess_period_ns(); })
      .def("last_run_timestamp",
           [](nvidia::gxf::PeriodicSchedulingTerm& pst) {
             auto result = pst.last_run_timestamp();
             if (!result) {
               throw std::runtime_error(GxfResultStr(result.error()));
             } else
               return result.value();
           })
      .def(
          "get",
          [](gxf_context_t context, gxf_uid_t cid, const char* name) {
            auto handle = nvidia::gxf::CreateHandleFromString<nvidia::gxf::PeriodicSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to PeriodicSchedulingTerm: %s", cid,
                            name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::BTSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "BTSchedulingTerm")
      .def(pybind11::init<>())
      .def("check",
           [](nvidia::gxf::BTSchedulingTerm& btst) {
             nvidia::gxf::SchedulingConditionType type;
             btst.check_abi(0, &type, 0);
             return type;
           })
      .def("set_condition",
           [](nvidia::gxf::BTSchedulingTerm& btst, nvidia::gxf::SchedulingConditionType* type) {
             btst.set_condition(*type);
             return;
           })
      .def(
          "get",
          [](gxf_context_t context, gxf_uid_t cid, const char* name) {
            auto handle = nvidia::gxf::CreateHandleFromString<nvidia::gxf::BTSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to BTSchedulingTerm: %s", cid, name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::TargetTimeSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "TargetTimeSchedulingTerm")
      .def(pybind11::init<>())
      .def("check",
           [](nvidia::gxf::TargetTimeSchedulingTerm& ttst, int64_t timestamp) {
             nvidia::gxf::SchedulingConditionType type;
             int64_t target_timestamp;
             ttst.check_abi(timestamp, &type, &target_timestamp);
             return type;
           })
      .def("set_next_target_time",
           [](nvidia::gxf::TargetTimeSchedulingTerm& ttst, int64_t target_timestamp) {
             auto result = ttst.setNextTargetTime(target_timestamp);
             if(result != GXF_SUCCESS){
              GXF_LOG_ERROR("[C%05zu] Couldn't set the target timestamp", ttst.cid());
              throw std::runtime_error(GxfResultStr(result));
             }
             return;
           })
      .def(
          "get",
          [](gxf_context_t context, gxf_uid_t cid, const char* name) {
            auto handle = nvidia::gxf::CreateHandleFromString<nvidia::gxf::TargetTimeSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to TargetTimeSchedulingTerm: %s", cid,
                            name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);
}
