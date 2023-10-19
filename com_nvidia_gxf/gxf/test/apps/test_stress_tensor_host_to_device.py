"""
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
This stress script generates yaml file which has 100 brodcast and gather
components. StepCount codelet is used to validate expected number of
count.
'''
def create_brodcast_gather(i, rep):
    return """---
name: copier_{i}
components:
- name: input
  type: nvidia::gxf::DoubleBufferReceiver
- name: output
  type: nvidia::gxf::DoubleBufferTransmitter
- name: allocator
  type: nvidia::gxf::UnboundedAllocator
- type: nvidia::gxf::TensorCopier
  parameters:
    receiver: input
    transmitter: output
    allocator: allocator
    mode: 0
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: input
    min_size: 1
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: output
    min_size: 1
---
name: generator_host_{i}
components:
- name: output
  type: nvidia::gxf::DoubleBufferTransmitter
- name: allocator
  type: nvidia::gxf::UnboundedAllocator
- type: nvidia::gxf::test::TensorGenerator
  parameters:
    output: output
    allocator: allocator
    shape: [ 4, 4, 4, 4 ]
    storage_type: 0
    enable_timestamps: false
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: output
    min_size: 1
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: 10000
---
name: generator_device_{i}
components:
- name: output
  type: nvidia::gxf::DoubleBufferTransmitter
- name: allocator
  type: nvidia::gxf::UnboundedAllocator
- type: nvidia::gxf::test::TensorGenerator
  parameters:
    output: output
    allocator: allocator
    shape: [ 4, 4, 4, 4 ]
    storage_type: 1
    enable_timestamps: false
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: output
    min_size: 1
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: 10000
---
name: comparator_{i}
components:
- name: expected
  type: nvidia::gxf::DoubleBufferReceiver
- name: actual
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::test::TensorComparator
  parameters:
    expected: expected
    actual: actual
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: expected
    min_size: 1
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: actual
    min_size: 1
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: 10000
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: generator_host_{i}/output
    target: copier_{i}/input
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: copier_{i}/output
    target: comparator_{i}/actual
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: generator_device_{i}/output
    target: comparator_{i}/expected
""".format(i = str(i), j = str(i), rep = rep,
           DownstreamReceptiveSchedulingTerm = "nvidia::gxf::DownstreamReceptiveSchedulingTerm")

def create_app(n):
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

  return header + "".join([create_brodcast_gather(i, 1000) for i in range(n)]) + footer

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_tensor_host_to_device_gen.yaml', create_app(100))
