/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_SAMPLE_PING_TX_HPP_
#define NVIDIA_GXF_SAMPLE_PING_TX_HPP_

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/resources.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Sample codelet implementation to send an entity
class PingTx : public Codelet {
 public:
  virtual ~PingTx() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

 private:
  Parameter<Handle<Transmitter>> signal_;
  Parameter<Handle<Clock>> clock_;
  Parameter<int64_t> trigger_interrupt_after_ms_;
  Resource<Handle<GPUDevice>> gpu_device_;
  int count = 1;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SAMPLE_PING_TX_HPP_
