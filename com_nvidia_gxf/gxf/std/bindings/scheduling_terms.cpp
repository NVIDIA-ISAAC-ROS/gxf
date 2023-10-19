/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/std/scheduling_terms.hpp"
#include "gxf/core/component.hpp"

template <typename S>
nvidia::gxf::Expected<nvidia::gxf::Handle<S>> getHandle(gxf_context_t context, gxf_uid_t cid, const char* name) {
  gxf_uid_t eid;
  std::string component_name;
  const std::string tag = std::string(name);
  const size_t pos = tag.find('/');

  if (pos == std::string::npos) {
    // Get the entity of this component
    const gxf_result_t result_1 = GxfComponentEntity(context, cid, &eid);
    if (result_1 != GXF_SUCCESS) {
      GXF_LOG_ERROR("%s", GxfResultStr(result_1));
      throw std::runtime_error(GxfResultStr(result_1));
    }
    component_name = tag;
  } else {
    // Split the tag into entity and component name
    const std::string entity_name = tag.substr(0, pos);
    component_name = tag.substr(pos + 1);
    // Search for the entity
    const gxf_result_t result_1 = GxfEntityFind(context, entity_name.c_str(), &eid);
    if (result_1 != GXF_SUCCESS) {
      GXF_LOG_ERROR(
          "[E%05zu] Could not find entity '%s' while parsing parameter '%s' of component %zu",
          eid, entity_name.c_str(), tag.c_str(), cid);
      throw std::runtime_error(GxfResultStr(result_1));
    }
  }
  // Get the type id of the component we are are looking for.
  gxf_tid_t tid;
  const gxf_result_t result_1 = GxfComponentTypeId(context, nvidia::TypenameAsString<S>(), &tid);
  if (result_1 != GXF_SUCCESS) {
    GXF_LOG_ERROR("%s", GxfResultStr(result_1));
    throw std::runtime_error(GxfResultStr(result_1));
  }
  gxf_uid_t cid2;
  // Find the component in the indicated entity
  const gxf_result_t result_2 =
      GxfComponentFind(context, eid, tid, component_name.c_str(), nullptr, &cid2);
  if (result_2 != GXF_SUCCESS) {
    GXF_LOG_ERROR("%s", GxfResultStr(result_2));
    throw std::runtime_error(GxfResultStr(result_2));
  }
  auto handle = nvidia::gxf::Handle<S>::Create(context, cid2);
  return handle;
}


PYBIND11_MODULE(scheduling_terms_pybind, m) {
  pybind11::class_<nvidia::gxf::SchedulingTerm>(m, "SchedulingTerm");
  pybind11::class_<nvidia::gxf::BooleanSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "BooleanSchedulingTerm")
      .def("make_enable_tick", [](nvidia::gxf::BooleanSchedulingTerm& bst) { bst.enable_tick(); })

      .def("make_disable_tick", [](nvidia::gxf::BooleanSchedulingTerm& bst) { bst.disable_tick(); })

      .def("check_tick_enable",
           [](nvidia::gxf::BooleanSchedulingTerm& bst) { return bst.checkTickEnabled(); })
      .def(
          "get",
          [](gxf_context_t context, gxf_uid_t cid, const char* name) {
            auto handle = getHandle<nvidia::gxf::BooleanSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to BooleanSchedulingTerm: %s", cid,
                            name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::CountSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "CountSchedulingTerm")
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
            auto handle = getHandle<nvidia::gxf::CountSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to CountSchedulingTerm: %s", cid, name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::PeriodicSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "PeriodicSchedulingTerm")
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
            auto handle = getHandle<nvidia::gxf::PeriodicSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to PeriodicSchedulingTerm: %s", cid,
                            name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::BTSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "BTSchedulingTerm")
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
            auto handle = getHandle<nvidia::gxf::BTSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to BTSchedulingTerm: %s", cid, name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::TargetTimeSchedulingTerm, nvidia::gxf::SchedulingTerm>(m, "TargetTimeSchedulingTerm")
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
            auto handle = getHandle<nvidia::gxf::TargetTimeSchedulingTerm>(context, cid, name);
            if (!handle) {
              GXF_LOG_ERROR("[C%05zu] Couldn't get a handle to TargetTimeSchedulingTerm: %s", cid,
                            name);
              throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
            }
            return handle.value().get();
          },
          pybind11::return_value_policy::reference);
}
