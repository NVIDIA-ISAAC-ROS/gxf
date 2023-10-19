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

#include "gxf/std/scheduling_condition.hpp"

PYBIND11_MODULE(scheduling_condition_pybind, m) {
  pybind11::enum_<nvidia::gxf::SchedulingConditionType>(m, "SchedulingConditionType")
      .value("NEVER", nvidia::gxf::SchedulingConditionType::NEVER)
      .value("READY", nvidia::gxf::SchedulingConditionType::READY)
      .value("WAIT", nvidia::gxf::SchedulingConditionType::WAIT)
      .value("WAIT_TIME", nvidia::gxf::SchedulingConditionType::WAIT_TIME)
      .value("WAIT_EVENT", nvidia::gxf::SchedulingConditionType::WAIT_EVENT);
}
