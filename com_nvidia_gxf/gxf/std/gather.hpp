/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_STD_GATHER_HPP
#define NVIDIA_GXF_STD_GATHER_HPP

#include <vector>

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
  gxf_result_t tick() override;

 private:
  Parameter<std::vector<Handle<Receiver>>> sources_;
  Parameter<Handle<Transmitter>> sink_;
  Parameter<int64_t> tick_source_limit_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
