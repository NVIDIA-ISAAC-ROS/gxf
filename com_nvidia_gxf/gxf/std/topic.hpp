/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_STD_TOPIC_HPP
#define NVIDIA_GXF_STD_TOPIC_HPP

#include <string>
#include <vector>

#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Class to add transmitters/receivers to a topic channel.
class Topic : public Component {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;

  // Get topic name.
  std::string getTopicName() const {
    return topic_name_.get().empty() ? std::string(name()) : topic_name_;
  }

  // Get all transmitters in this topic.
  std::vector<Handle<Transmitter>> getTransmitters() const { return transmitters_.get(); }

  // Get all receivers in this topic.
  std::vector<Handle<Receiver>> getReceivers() const { return receivers_.get(); }

 private:
  Parameter<std::string> topic_name_;
  Parameter<std::vector<Handle<Receiver>>> receivers_;
  Parameter<std::vector<Handle<Transmitter>>> transmitters_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
