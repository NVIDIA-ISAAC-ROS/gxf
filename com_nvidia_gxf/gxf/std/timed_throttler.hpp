/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_SERIALIZATION_TIMED_THROTTLER_HPP_
#define NVIDIA_GXF_SERIALIZATION_TIMED_THROTTLER_HPP_

#include <string>

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Publishes entities respecting the difference in acquisition times between subsequent entities.
// The entities received are generated based on a different clock (called the throttling clock).
// During the first tick, the start time is extracted from the throttling clock and the offset is
// used to map the acqtime and pubtime of the entities from the throttling clock to the execution
// clock. During every tick, the codelet peeks into the next entity in the queue and then asks the
// scheduler to be executed at the acqtime of the next entity.
class TimedThrottler : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  Parameter<Handle<Clock>> execution_clock_;
  Parameter<Handle<Clock>> throttling_clock_;
  Parameter<Handle<Receiver>> receiver_;
  Parameter<Handle<Transmitter>> transmitter_;
  Parameter<Handle<TargetTimeSchedulingTerm>> scheduling_term_;

  // Store entity published at the next tick
  Expected<Entity> cached_entity_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  // Initial_time_offset of the publishing clock. The offset is stored as double to account for
  // both positive and negative offsets
  int64_t time_offset_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SERIALIZATION_TIMED_THROTTLER_HPP_
