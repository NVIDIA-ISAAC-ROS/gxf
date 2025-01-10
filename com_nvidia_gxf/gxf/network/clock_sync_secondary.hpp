/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NETWORK_CLOCK_SYNC_SECONDARY_HPP_
#define NVIDIA_GXF_NETWORK_CLOCK_SYNC_SECONDARY_HPP_

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/synthetic_clock.hpp"

namespace nvidia {
namespace gxf {

/**
 * @brief Sets application clock to received timestamp.
 *
 * This codelet is meant to subscribe to a timestamp message
 * sent from another application, to synchronize clocks between the two
 * applications by setting its application's clock directly.
 */
class ClockSyncSecondary : public Codelet {
 public:
    gxf_result_t registerInterface(Registrar* registrar) override;
    gxf_result_t tick() override;
 private:
    // A handle to the receiver which subscribes to the timestamp
    Parameter<Handle<Receiver>> rx_timestamp_;
    // A handle to the application clock
    Parameter<Handle<SyntheticClock>> synthetic_clock_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_NETWORK_CLOCK_SYNC_SECONDARY_HPP_
