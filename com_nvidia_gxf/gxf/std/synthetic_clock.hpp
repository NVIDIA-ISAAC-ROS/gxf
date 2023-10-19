/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_SYNTHETIC_CLOCK_HPP_
#define NVIDIA_GXF_STD_SYNTHETIC_CLOCK_HPP_

#include <condition_variable>
#include <mutex>

#include "gxf/std/clock.hpp"

namespace nvidia {
namespace gxf {

/// @brief A clock where time flow is synthesized, like from a recording or a simulation
class SyntheticClock : public gxf::Clock {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  double time() const override;
  int64_t timestamp() const override;
  gxf::Expected<void> sleepFor(int64_t duration_ns) override;
  gxf::Expected<void> sleepUntil(int64_t target_time_ns) override;

  // manually advance the clock to a desired new target time
  gxf::Expected<void> advanceTo(int64_t new_time_ns);

  // manually advance the clock by a given delta
  gxf::Expected<void> advanceBy(int64_t time_delta_ns);

 private:
  gxf::Parameter<int64_t> initial_timestamp_;

  int64_t current_time_;

  std::mutex mutex_;
  std::condition_variable condition_variable_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_SYNTHETIC_CLOCK_HPP_
