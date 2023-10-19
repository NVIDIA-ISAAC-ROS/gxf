/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/serialization/tests/serialization_tester.hpp"

#include <vector>

namespace nvidia {
namespace gxf {
namespace test {

gxf_result_t SerializationTester::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(input_, "input");
  result &= registrar->parameter(output_, "output");
  result &= registrar->parameter(entity_serializer_, "entity_serializer");
  result &= registrar->parameter(serialization_buffer_, "serialization_buffer");
  return ToResultCode(result);
}

gxf_result_t SerializationTester::tick() {
  serialization_buffer_->reset();
  return ToResultCode(
    input_->receive()
    .map([&](Entity entity) {
      return entity_serializer_->serializeEntity(entity, serialization_buffer_.get());
    })
    .and_then([&]() {
      return Entity::New(context());
    })
    .map([&](Entity entity) {
      return ExpectedOrError(
          entity_serializer_->deserializeEntity(entity, serialization_buffer_.get()), entity);
    })
    .map([&](Entity entity) { return output_->publish(entity); }));
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
