/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/test/components/mock_receiver.hpp"

namespace nvidia {
namespace gxf {
namespace test {

gxf_result_t MockReceiver::registerInterface(Registrar* registrar) {
  if (registrar == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  Expected<void> result;
  result &= registrar->parameter(
      ignore_unreceived_entities_, "ignore_unreceived_entities", "Ignore Unreceived Entities",
      "Does not raise an error if there are unreceived entities in the buffer",
      false);
  result &= registrar->parameter(
      fail_on_receive_, "fail_on_receive", "Fail On Receive",
      "Forces receive_abi() to return GXF_FAILURE",
      false);
  result &= registrar->parameter(
      fail_on_sync_, "fail_on_sync", "Fail On Sync",
      "Forces sync_abi() to return GXF_FAILURE",
      false);
  result &= registrar->parameter(
      max_capacity_, "max_capacity", "Max Capacity",
      "Maximum number of entities that can be buffered at once",
      entities_.max_size());
  return ToResultCode(result);
}

gxf_result_t MockReceiver::initialize() {
  entities_.clear();
  metrics_ = Metrics{0, 0, 0};
  return GXF_SUCCESS;
}

gxf_result_t MockReceiver::deinitialize() {
  printMetrics();
  auto result = checkForUnreceivedEntities();
  if (!ignore_unreceived_entities_ && !result) {
    return ToResultCode(result);
  }
  return GXF_SUCCESS;
}

gxf_result_t MockReceiver::pop_abi(gxf_uid_t* uid) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (uid == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (entities_.empty()) {
    return GXF_FAILURE;
  }
  const Entity entity = entities_.front();
  entities_.pop_front();
  const gxf_result_t code = GxfEntityRefCountInc(context(), entity.eid());
  if (code != GXF_SUCCESS) {
    return code;
  }
  *uid = entity.eid();
  return GXF_SUCCESS;
}

gxf_result_t MockReceiver::push_abi(gxf_uid_t other) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (entities_.size() >= capacity()) {
    return GXF_EXCEEDING_PREALLOCATED_SIZE;
  }
  return ToResultCode(
      Entity::Shared(context(), other)
      .map([&](Entity entity) {
        entities_.push_back(entity);
        if (entities_.size() > metrics_.peak) {
          metrics_.peak = entities_.size();
        }
        return Success;
      }));
}

gxf_result_t MockReceiver::peek_abi(gxf_uid_t* uid, int32_t index) {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (uid == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (static_cast<size_t>(index) >= entities_.size()) {
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  *uid = entities_[index].eid();
  return GXF_SUCCESS;
}

size_t MockReceiver::size_abi() {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return entities_.size();
}

gxf_result_t MockReceiver::receive_abi(gxf_uid_t* uid) {
  if (fail_on_receive_) {
    return GXF_FAILURE;
  }
  const gxf_result_t code = pop_abi(uid);
  if (code != GXF_SUCCESS) {
    return code;
  }
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  metrics_.received++;
  return GXF_SUCCESS;
}

gxf_result_t MockReceiver::sync_abi() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (fail_on_sync_) {
    return GXF_FAILURE;
  }
  metrics_.syncs++;
  return GXF_SUCCESS;
}

gxf_result_t MockReceiver::sync_io_abi() {
  return GXF_SUCCESS;
}

Expected<void> MockReceiver::checkForUnreceivedEntities() {
  if (!entities_.empty()) {
    GXF_LOG_WARNING("[%s/%s] Unreceived Entities: %zu", entity().name(), name(), entities_.size());
    return Unexpected{GXF_FAILURE};
  }
  return Success;
}

void MockReceiver::printMetrics() {
  GXF_LOG_INFO("[%s/%s] Entities Received: %zu { Syncs: %zu | Peak: %zu }",
               entity().name(), name(), metrics_.received, metrics_.syncs, metrics_.peak);
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
