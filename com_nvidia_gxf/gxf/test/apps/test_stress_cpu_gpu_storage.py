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

# This stress script generates yaml file which publishes and receives
# CPU, GPU bound memory messages across 100 transmitter and receiver.
# StepCount codelet is used to validate expected number of count.
def create_rx_tx(i, count, rep):
  return """---
name: tx_{i}
components:
- name: c0
  type: nvidia::gxf::DoubleBufferTransmitter
- name: pool
  type: nvidia::gxf::test::MockAllocator
- type: nvidia::gxf::test::IntegerSinSum
  parameters:
    count: {count}
    result: c0
    pool: pool
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: {rep}
- name: tensor
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: tensor
    min_size: 1
- name: pool1
  type: nvidia::gxf::test::MockAllocator
- type: nvidia::gxf::NppiSet
  parameters:
    rows: 320
    columns: 480
    channels: 3
    pool: pool1
    value: [-0.54, 0.21, 0.73]
    out: tensor
---
name: rx_{j}
components:
- name: c0
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: c0
    min_size: 1
- type: nvidia::gxf::test::Pop
  parameters:
    message: c0
- name: tensorGPU
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: tensorGPU
    min_size: 1
- type: nvidia::gxf::test::PrintTensor
  parameters:
    tensors: tensorGPU
    silent: True
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: {rep}
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx_{i}/c0
    target: rx_{j}/c0
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx_{i}/tensor
    target: rx_{j}/tensorGPU
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

  return header + "".join([create_rx_tx(i, burn, 1000) for i in range(n)]) + footer

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_cpu_gpu_storage_gen.yaml', create_app(100, 5000000))