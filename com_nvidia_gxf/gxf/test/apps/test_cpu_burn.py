"""
 SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import yaml
import sys


def create_rx_tx(i, count, rep):
  return """---
name: tx_{i}
components:
- name: {i}
  type: nvidia::gxf::test::MockTransmitter
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: {i}
    min_size: 1
- name: pool
  type: nvidia::gxf::test::MockAllocator
- type: nvidia::gxf::test::IntegerSinSum
  parameters:
    count: {count}
    result: {i}
    pool: pool
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: {rep}
---
name: rx_{j}
components:
- name: {j}
  type: nvidia::gxf::test::MockReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: {j}
    min_size: 1
- type: nvidia::gxf::test::Pop
  parameters:
    message: {j}
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: {rep}
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx_{i}/{i}
    target: rx_{j}/{j}
""".format(i = "c" + str(i), j = "c" + str(i), count = count, rep = rep)

def create_app(n, burn):
  header = """%YAML 1.2
"""

  footer = """---
components:
- type: nvidia::gxf::GreedyScheduler
  parameters:
    max_duration_ms: 1000000
    clock: clock
- name: clock
  type: nvidia::gxf::ManualClock
"""

  return header + "".join([create_rx_tx(i, burn, 10) for i in range(n)]) + footer


def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)


if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_cpu_burn_1.yaml', create_app(1, 5000000))
  save(out_dir + '/test_cpu_burn_3.yaml', create_app(3, 5000000))
  save(out_dir + '/test_cpu_burn_10.yaml', create_app(10, 1000000))
  save(out_dir + '/test_cpu_burn_25.yaml', create_app(25, 500000))
  save(out_dir + '/test_cpu_burn_100.yaml', create_app(100, 100000))
