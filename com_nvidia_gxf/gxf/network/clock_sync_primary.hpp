/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NETWORK_CLOCK_SYNC_PRIMARY_HPP_
#define NVIDIA_GXF_NETWORK_CLOCK_SYNC_PRIMARY_HPP_

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

/**
 * @brief Publishes graph's current clock time to transmitter.
 *
 * This codelet is meant to publish current graph time across an interface
 * to another application, which will then process this graph's time as
 * part of its operation. For example, the client app may set its clock
 * to the time published by this codelet (see: ClockSyncSecondary).
 */
class ClockSyncPrimary : public Codelet {
 public:
      gxf_result_t registerInterface(Registrar* registrar) override;
      gxf_result_t start() override { return GXF_SUCCESS; }
      gxf_result_t tick() override;
      gxf_result_t stop() override { return GXF_SUCCESS; }
 private:
      // A handle to the transmitter which will publish the timestamp
      Parameter<Handle<Transmitter>> tx_timestamp_;
      // A handle to the application clock
      Parameter<Handle<Clock>> clock_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_NETWORK_CLOCK_SYNC_PRIMARY_HPP_
