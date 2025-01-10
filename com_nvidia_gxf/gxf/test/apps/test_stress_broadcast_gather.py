"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


'''
This stress script generates yaml file which has 100 broadcast and gather
components. StepCount codelet is used to validate expected number of
count.
'''
def create_broadcast_gather(i, rep):
    return """---
name: tx_{i}
components:
- name: tensor
  type: nvidia::gxf::DoubleBufferTransmitter
  parameters:
    capacity: 3
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: tensor
    min_size: 1
- name: pool
  type: nvidia::gxf::BlockMemoryPool
  parameters:
    storage_type: 1
    block_size: 1843200
    num_blocks: 6
- type: nvidia::gxf::NppiSet
  parameters:
    rows: 320
    columns: 480
    channels: 3
    pool: pool
    value: [-0.54, 0.21, 0.73]
    out: tensor
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: {rep}
---
name: broadcast_{i}
components:
- name: source
  type: nvidia::gxf::DoubleBufferReceiver
  parameters:
    capacity: 3
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: source
    min_size: 1
- name: ping_1
  type: nvidia::gxf::DoubleBufferTransmitter
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: ping_1
    min_size: 1
- name: ping_2
  type: nvidia::gxf::DoubleBufferTransmitter
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: ping_2
    min_size: 1
- name: ping_3
  type: nvidia::gxf::DoubleBufferTransmitter
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: ping_3
    min_size: 1
- type: nvidia::gxf::Broadcast
  parameters:
    source: source
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: {rep}
    use_assert: true
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx_{i}/tensor
    target: broadcast_{i}/source
---
name: gather_{i}
components:
- name: input_1
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: input_1
    min_size: 1
- name: input_2
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: input_2
    min_size: 1
- name: input_3
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: input_3
    min_size: 1
- name: output
  type: nvidia::gxf::DoubleBufferTransmitter
  parameters:
    capacity: 3
- type: nvidia::gxf::Gather
  parameters:
    sink: output
    tick_source_limit: 1
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: output
    min_size: 3
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: {rep}
    use_assert: true
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: broadcast_{i}/ping_1
    target: gather_{i}/input_1
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: broadcast_{i}/ping_2
    target: gather_{i}/input_2
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: broadcast_{i}/ping_3
    target: gather_{i}/input_3
---
name: rx_{i}
components:
- name: signal
  type: nvidia::gxf::DoubleBufferReceiver
  parameters:
    capacity: 3
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: signal
    min_size: 1
- type: nvidia::gxf::PingRx
  parameters:
    signal: signal
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: gather_{i}/output
    target: rx_{i}/signal
""".format(i = "c" + str(i), j = "c" + str(i), rep = rep,
           DownstreamReceptiveSchedulingTerm = "nvidia::gxf::DownstreamReceptiveSchedulingTerm")

HEADER = """%YAML 1.2
"""

GREEDY_FOOTER = """---
components:
- type: nvidia::gxf::GreedyScheduler
  parameters:
    max_duration_ms: 1000000
    clock: clock
- name: clock
  type: nvidia::gxf::ManualClock
"""

MULTITHREAD_FOOTER = """---
components:
- type: nvidia::gxf::MultiThreadScheduler
  parameters:
    max_duration_ms: 100000
    clock: clock
    worker_thread_number: 5
    stop_on_deadlock: false
- name: clock
  type: nvidia::gxf::RealtimeClock
"""

def create_greedy_app(n):
  return HEADER + "".join([create_broadcast_gather(i, 1000) for i in range(n)]) + GREEDY_FOOTER

def create_multithread_app(n):
  return (
    HEADER + "".join([create_broadcast_gather(i, 1000) for i in range(n)]) + MULTITHREAD_FOOTER
  )

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_broadcast_gather_greedy_gen.yaml', create_greedy_app(100))
  save(out_dir + '/test_stress_broadcast_gather_multithread_gen.yaml', create_multithread_app(1))
