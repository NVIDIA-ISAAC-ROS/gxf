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
#ifndef NVIDIA_GXF_TEST_MESSAGE_HELPER_HPP_
#define NVIDIA_GXF_TEST_MESSAGE_HELPER_HPP_

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/component.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/tensor.hpp"
#include "gxf/std/typed_message_view.hpp"

namespace nvidia {
namespace gxf {

namespace my_formats {

  nvidia::gxf::TypedMessageView<Tensor, Tensor> test_format_2("T1", "T2");

}  // my_formats (used in test_typed_message_view.cpp)

}  // namespace gxf
}  // namespace nvidia

#endif
