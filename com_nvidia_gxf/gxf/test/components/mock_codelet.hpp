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
#ifndef NVIDIA_GXF_TEST_COMPONENTS_MOCK_CODELET_HPP_
#define NVIDIA_GXF_TEST_COMPONENTS_MOCK_CODELET_HPP_

#include <memory>
#include <random>
#include <string>

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {
namespace test {

class Frame : public Component {
 public:
  int64_t create_time;
  int64_t first_process_time = -1;
  int64_t process_duration;
  int frame_id;
  int num_processed;
  bool end = false;;
};

class MockCodelet : public Codelet {
 public:
  virtual ~MockCodelet() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }
  static const char* const kSRC;
  static const char* const kSINK;
  static const char* const kPROCESS;
  static const char* const kLATENCY;

 private:
  Parameter<std::string> mode_;
  Parameter<Handle<Receiver>> receiver_;
  Parameter<Handle<Transmitter>> transmitter_;
  Parameter<Handle<Clock>> clock_;
  Parameter<int> lower_;
  Parameter<int> upper_;
  Parameter<bool> include_mock_latency_;
  std::unique_ptr<std::uniform_int_distribution<int>> uniform_dist_;
  std::unique_ptr<std::random_device> rd_;
  std::unique_ptr<std::mt19937> gen_;
  int receive_count_ = 0;
  int transmit_count_ = 0;

 private:
  gxf_result_t src_mode();
  gxf_result_t sink_mode();
  gxf_result_t process_mode();
  double ms(const int64_t& ns);
  int64_t now();
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_COMPONENTS_MOCK_CODELET_HPP_
