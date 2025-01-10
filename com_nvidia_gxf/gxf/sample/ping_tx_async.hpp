/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_GXF_SAMPLE_PING_TX_ASYNC_HPP_
#define NVIDIA_GXF_SAMPLE_PING_TX_ASYNC_HPP_

#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

/**
 * @brief Sample codelet implementation to send an entity asynchronously. This is used in GXF
 * Asynchronous Lock-free Buffer for testing.
 *
 */
class PingTxAsync : public Codelet {
 public:
  virtual ~PingTxAsync() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  Parameter<Handle<Transmitter>> signal_;
  Parameter<int64_t> sleep_time_us_;
  int count = 0;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SAMPLE_PING_TX_ASYNC_HPP_
