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
#include "gxf/benchmark/tests/async_trigger_data_replayer.hpp"

#include <algorithm>
#include <vector>

#include "gxf/benchmark/gems/data_replay_control.hpp"
#include "gxf/core/expected_macro.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {
namespace benchmark {
namespace test {

gxf_result_t AsyncTriggerDataReplayer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      receiver_, "receiver",
      "Replay control receiver",
      "A receiver for receiving replay control messages");
  result &= registrar->parameter(
      replay_data_source_async_scheduling_term_,
      "replay_data_source_async_scheduling_term",
      "Replay data source async scheduling terms",
      "Scheduling terms to control execution of replay data source");
  return gxf::ToResultCode(result);
}

gxf_result_t AsyncTriggerDataReplayer::start() {
  replay_data_source_async_scheduling_term_.get()->setEventState(
      nvidia::gxf::AsynchronousEventState::WAIT);
  return GXF_SUCCESS;
}

gxf_result_t AsyncTriggerDataReplayer::tick() {
  auto message = receiver_->receive();
  if (!message) {
    return message.error();
  }

  auto replay_command = message->get<DataReplayControl>().value()->replay_command;
  if (replay_command == DataReplayControl::Command::kPlay) {
    replay_data_source_async_scheduling_term_.get()->setEventState(
        nvidia::gxf::AsynchronousEventState::EVENT_DONE);
  } else if (replay_command == DataReplayControl::Command::kPause) {
    replay_data_source_async_scheduling_term_.get()->setEventState(
        nvidia::gxf::AsynchronousEventState::WAIT);
  } else {
    GXF_LOG_ERROR("Unsupported replay command: %d", static_cast<int>(replay_command));
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
