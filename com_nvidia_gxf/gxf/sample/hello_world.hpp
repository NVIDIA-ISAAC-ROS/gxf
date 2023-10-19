/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_SAMPLE_HELLO_WORLD_HPP_
#define NVIDIA_GXF_SAMPLE_HELLO_WORLD_HPP_

#include "gxf/std/codelet.hpp"

namespace nvidia {
namespace gxf {

// Sample codelet implementation to print 'Hello World' on every tick
class HelloWorld : public Codelet {
 public:
  virtual ~HelloWorld() = default;

  gxf_result_t start() override { return GXF_SUCCESS; }

  gxf_result_t tick() override {
    GXF_LOG_INFO("Hello world");
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override { return GXF_SUCCESS; }
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_SAMPLE_HELLO_WORLD_HPP_
