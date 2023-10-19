/*
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/tests/test_clock_sync_helpers.hpp"

#include "gxf/std/parameter_parser_std.hpp"

namespace nvidia {
namespace gxf {
namespace test {

gxf_result_t ClockTester::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(synthetic_clock_, "synthetic_clock");
  result &= registrar->parameter(boolean_scheduling_term_, "boolean_scheduling_term");
  result &= registrar->parameter(timestamps_, "timestamps");
  return ToResultCode(result);
}

gxf_result_t ClockTester::start() {
  boolean_scheduling_term_->enable_tick();
  iter_ = timestamps_.get().begin();
  return GXF_SUCCESS;
}

gxf_result_t ClockSetter::tick() {
  auto result = synthetic_clock_->advanceTo(*iter_);
  if (!result) {
    GXF_LOG_ERROR("synthetic_clock_->advanceTo() failed for timestamp %ldns.", *iter_);
    return GXF_FAILURE;
  }
  GXF_LOG_DEBUG("Updated synthetic clock to %ldns.", *iter_);
  iter_++;
  if (iter_ == timestamps_.get().end()) {
    boolean_scheduling_term_->disable_tick();
  }
  return GXF_SUCCESS;
}

gxf_result_t ClockChecker::tick() {
  GXF_LOG_INFO("tick - checker.");
  if (iter_ == timestamps_.get().end()) {
    boolean_scheduling_term_->disable_tick();
    return GXF_SUCCESS;
  }
  if (synthetic_clock_->timestamp() != *iter_++) {
    // fail if clock timestamp does not match user input
    const auto pv = std::prev(iter_, 1);
    GXF_LOG_ERROR("synthetic_clock_->timestamp() did not match expected %ldns.", *pv);
    return GXF_FAILURE;
  }
  GXF_LOG_INFO("Synthetic clock matched expected.");
  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
