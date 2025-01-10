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
#ifndef NVIDIA_GXF_TEST_EXTENSIONS_TEST_LOAD_EXTENSION_HPP_
#define NVIDIA_GXF_TEST_EXTENSIONS_TEST_LOAD_EXTENSION_HPP_

#include "gxf/std/codelet.hpp"

namespace nvidia {
namespace gxf {
namespace test {

class LoadExtensionFromPointerTest : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(value_, "value");
    return ToResultCode(result);
  }
  gxf_result_t tick() override {
    return GXF_SUCCESS;
  }

  int32_t value() const { return value_; }

  void value(int32_t value) {
    value_.set(value);
  }

 private:
  Parameter<int32_t> value_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_EXTENSIONS_TEST_LOAD_EXTENSION_HPP_
