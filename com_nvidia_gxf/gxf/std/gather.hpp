/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_GATHER_HPP
#define NVIDIA_GXF_STD_GATHER_HPP

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// All messages arriving on any input channel are published on the single output channel. This
// component automatically uses all Receiver components which are on the same entity.
class Gather : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;

  gxf_result_t start() override;

  gxf_result_t tick() override;

  gxf_result_t stop() override;

 private:
  Parameter<Handle<Transmitter>> sink_;
  Parameter<int64_t> tick_source_limit_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
