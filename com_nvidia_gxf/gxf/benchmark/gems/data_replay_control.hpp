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
#pragma once

#include <cstdint>

namespace nvidia {
namespace gxf {
namespace benchmark {

// Message component used to control data replay during runtime
struct DataReplayControl {
  enum class Command {
    kPlay = 0,   // Resumes the replay stream
    kPause = 1,  // Pauses the replay stream
    kStep = 2,   // Advances the replay stream forward by 1 message
    kSeek = 3,   // Repositions the replay stream to the given timestamp
  };

  // Replay command
  Command replay_command;
  // Seek timestamp in nanoseconds
  int64_t seek_timestamp;
};

}  // namespace benchmark
}  // namespace gxf
}  // namespace nvidia
