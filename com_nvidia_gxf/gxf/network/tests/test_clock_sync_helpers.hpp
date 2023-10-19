/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NETWORK_TEST_CLOCK_SYNC_HELPERS_HPP_
#define NVIDIA_GXF_NETWORK_TEST_CLOCK_SYNC_HELPERS_HPP_

#include <vector>

#include "gxf/std/codelet.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/synthetic_clock.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Sets SyntheticClock time based on user input
class ClockTester : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override { return GXF_SUCCESS; }
  gxf_result_t deinitialize() override { return GXF_SUCCESS; }

  gxf_result_t start() override;
  gxf_result_t tick() override = 0;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 protected:
  Parameter<Handle<SyntheticClock>> synthetic_clock_;
  Parameter<Handle<BooleanSchedulingTerm>> boolean_scheduling_term_;
  Parameter<std::vector<int64_t>> timestamps_;

  std::vector<int64_t>::const_iterator iter_;
};

// Sets provided clock to user-provided timestamps on tick()
class ClockSetter : public ClockTester {
 public:
  gxf_result_t tick() override;
};


// Checks if provided clock matches user-provided timestamps on tick()
class ClockChecker : public ClockTester {
 public:
  gxf_result_t tick() override;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_NETWORK_TEST_CLOCK_SYNC_HELPERS_HPP_
