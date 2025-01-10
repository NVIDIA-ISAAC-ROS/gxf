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
#ifndef NVIDIA_GXF_SAMPLE_PING_RX_HPP_
#define NVIDIA_GXF_SAMPLE_PING_RX_HPP_

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/resources.hpp"

namespace nvidia {
namespace gxf {

// Sample codelet implementation to receive an entity
class PingRx : public Codelet {
 public:
  virtual ~PingRx() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

 private:
  Parameter<Handle<Receiver>> signal_;
  Resource<Handle<GPUDevice>> gpu_device_;
  int count = 1;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SAMPLE_PING_RX_HPP_
