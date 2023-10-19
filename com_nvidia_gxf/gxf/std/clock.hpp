/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NVIDIA_GXF_STD_CLOCK_HPP_
#define NVIDIA_GXF_STD_CLOCK_HPP_

#include <chrono>

#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {

/// @brief Keeps track of time
//
// This clock is based on a steady clock however time can be scaled to run slower or faster.
class Clock : public Component {
 public:
  virtual ~Clock() = default;

  /// @brief The current time of the clock. Time is measured in seconds.
  virtual double time() const = 0;

  /// @brief The current timestamp of the clock. Timestamps are measured in nanoseconds.
  virtual int64_t timestamp() const = 0;

  /// @brief Waits until the given duration has elapsed on the clock
  virtual Expected<void> sleepFor(int64_t duration_ns) = 0;

  /// @brief Waits until the given target time
  virtual Expected<void> sleepUntil(int64_t target_ns) = 0;
};

/// @brief A clock which runs based on a realtime clock
class RealtimeClock : public Clock {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  double time() const override;
  int64_t timestamp() const override;
  Expected<void> sleepFor(int64_t duration_ns) override;
  Expected<void> sleepUntil(int64_t target_time_ns) override;

  // Changes time scaling used by the clock.
  Expected<void> setTimeScale(double time_scale);

 private:
  Parameter<double> initial_time_offset_;
  Parameter<double> initial_time_scale_;
  Parameter<bool> use_time_since_epoch_;

  std::chrono::time_point<std::chrono::steady_clock> reference_;
  double time_offset_;
  double time_scale_;
};

/// @brief A clock where time flow is controlled manually
class ManualClock : public Clock {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  double time() const override;
  int64_t timestamp() const override;
  Expected<void> sleepFor(int64_t duration_ns) override;
  Expected<void> sleepUntil(int64_t target_time_ns) override;

 private:
  Parameter<int64_t> initial_timestamp_;

  int64_t current_time_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
