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

#include <memory>
#include <string>
#include <thread>
#include "gxf/test/components/mock_codelet.hpp"

namespace nvidia {
namespace gxf {
namespace test {

const char* const MockCodelet::kSRC = "SRC";
const char* const MockCodelet::kSINK = "SINK";
const char* const MockCodelet::kPROCESS = "PROCESS";
const char* const MockCodelet::kLATENCY = "LATENCY";

gxf_result_t MockCodelet::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(mode_, "mode", "Mode",
      "Mode of this codelet", std::string(kPROCESS));
  result &= registrar->parameter(receiver_, "receiver", "Receiver",
      "Channel to receive messages from another graph entity",
       Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(transmitter_, "transmitter", "Transmitter",
      "channel to publish messages to other graph entities",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      clock_, "clock", "Clock", "Clock component needed for timestamping messages");
  result &= registrar->parameter(lower_, "duration_lower_ms",
      "random tick duration lower bound",
      "lower bound for uniform distribution of tick duration [ms]",
      -1);
  result &= registrar->parameter(upper_, "duration_upper_ms",
      "random tick duration upper bound",
      "upper bound for uniform distribution of tick duration [ms]",
      -1);
  result &= registrar->parameter(include_mock_latency_, "include_mock_latency",
      "total latency or scheduling latency",
      "whether or not to include tick duration in latency calculations",
      false);
  return ToResultCode(result);
}

gxf_result_t MockCodelet::start() {
  if (lower_.get() > upper_.get()) {
    GXF_LOG_ERROR("Codelet[%s]: Tick duration lower bound and upper bound misconfigured. "
                  "lower: %d, upper: %d", name(), lower_.get(), upper_.get());
    return GXF_FAILURE;
  }

  if (lower_.get() == -1 || upper_.get() == -1) {
    GXF_LOG_INFO("No sleep in stick()");
    return GXF_SUCCESS;
  }
  rd_ = std::make_unique<std::random_device>();
  gen_ = std::make_unique<std::mt19937>((*rd_)());
  uniform_dist_ = std::make_unique<std::uniform_int_distribution<int>>(lower_.get(), upper_.get());
  return GXF_SUCCESS;
}

gxf_result_t MockCodelet::tick() {
  if (mode_.get() == kSRC) {
    return src_mode();
  } else if (mode_.get() == kSINK) {
    return sink_mode();
  } else if (mode_.get() == kPROCESS) {
    return process_mode();
  }
  GXF_LOG_ERROR("Not supported work mode: %s", mode_.get().c_str());
  return GXF_FAILURE;
}

gxf_result_t MockCodelet::src_mode() {
  // create message
  auto message = Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failure creating message entity.");
    return message.error();
  }

  // mock create frame duration
  if (uniform_dist_ != nullptr && gen_ != nullptr) {
    // imagine sensor takes some time to create a frame
    int rand_num = (*uniform_dist_)(*gen_);
    GXF_LOG_DEBUG("Codelet[%s]: Mock tick execution: %d ms", name(), rand_num);
    std::this_thread::sleep_for(std::chrono::milliseconds(rand_num));
  }

  // get current time
  int64_t now = this->now();
  // add a frame to the message
  auto maybe_frame = message.value().add<Frame>(kLATENCY);
  if (!maybe_frame) { return ToResultCode(maybe_frame); }
  auto frame = maybe_frame.value();
  frame->create_time = now;
  frame->first_process_time = -1;
  frame->frame_id = ++transmit_count_;
  // send message
  auto transmitter = transmitter_.try_get();
  if (!transmitter) { return GXF_ARGUMENT_INVALID; }
  auto result = transmitter.value()->publish(message.value(), now);
  GXF_LOG_DEBUG("Codelet[%s]: Message Sent: %d", name(), transmit_count_);
  return ToResultCode(message);
}

gxf_result_t MockCodelet::sink_mode() {
  // get current time
  int64_t now = this->now();

  // receive message
  auto receiver = receiver_.try_get();
  if (!receiver) { return GXF_ARGUMENT_INVALID; }
  auto message = receiver.value()->receive();
  if (!message || message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
  // get the frame from message
  auto maybe_frame = message.value().get<Frame>(kLATENCY);
  if (!maybe_frame) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
  auto frame = maybe_frame.value();
  if (frame->end) {
    GXF_LOG_ERROR("Frame[%d] life cycle already ended", frame->frame_id);
    return GXF_FAILURE;
  }
  frame->end = true;

  // print latency info
  double latency;
  if (include_mock_latency_.get()) {
    latency = ms(now - frame->first_process_time);
  } else {
    latency = ms(now - frame->first_process_time - frame->process_duration);
  }
  (void)latency;  // avoid unused variable warning
  GXF_LOG_INFO("Codelet[%s]: frame: %d, latency: %.3f, end_time: %.4f, create_time: %.4f, "
               "process_duration: %.4f, first_process: %.4f, num_processed: %d",
               name(), frame->frame_id, latency, ms(now), ms(frame->create_time),
               ms(frame->process_duration), ms(frame->first_process_time), frame->num_processed);
  return GXF_SUCCESS;
}

gxf_result_t MockCodelet::process_mode() {
  // get start time
  int64_t start = this->now();

  // receive message
  auto receiver = receiver_.try_get();
  if (!receiver) { return GXF_ARGUMENT_INVALID; }
  auto message = receiver.value()->receive();
  GXF_LOG_DEBUG("Codelet[%s]: Message Received: %d", name(), ++receive_count_);
  if (!message || message.value().is_null()) {
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }

  // get the frame from message
  auto maybe_frame = message.value().get<Frame>(kLATENCY);
  if (!maybe_frame) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
  auto frame = maybe_frame.value();
  if (frame->end) {
    GXF_LOG_ERROR("Frame[%d] life cycle already ended", frame->frame_id);
    return GXF_FAILURE;
  }

  // mock execute
  if (uniform_dist_ != nullptr && gen_ != nullptr) {
    // imagine do something with frame
    int rand_num = (*uniform_dist_)(*gen_);
    GXF_LOG_DEBUG("Codelet[%s]: Mock tick execution: %d ms", name(), rand_num);
    std::this_thread::sleep_for(std::chrono::milliseconds(rand_num));
  }

  // transmit message
  auto transmitter = transmitter_.try_get();
  if (!transmitter) { return GXF_ARGUMENT_INVALID; }
  // get end time
  int64_t end = this->now();

  // add tick duration to this frame
  frame->process_duration += end - start;
  frame->num_processed++;
  // if this codelet is the first process codelet, set first process time
  if (frame->first_process_time == -1) {
    frame->first_process_time = start;
  }
  auto result = transmitter.value()->publish(message.value(), end);
  GXF_LOG_DEBUG("Codelet[%s]: Message Sent: %d", name(), ++transmit_count_);
  return GXF_SUCCESS;
}

double MockCodelet::ms(const int64_t& ns) {
  return static_cast<double>(ns) / 1000000.0;
}

int64_t MockCodelet::now() {
  auto maybe_clock = clock_.try_get();
  if (!maybe_clock) { return GXF_ARGUMENT_INVALID; }
  return maybe_clock.value()->timestamp();
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
