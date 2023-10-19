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
def create_rx_tx(i):
  return """---
name: stream_tensor_generator_{i}
components:
- name: host_out
  type: nvidia::gxf::DoubleBufferTransmitter
- name: cuda_out
  type: nvidia::gxf::DoubleBufferTransmitter
- name: generator
  type: nvidia::gxf::test::cuda::StreamTensorGenerator
  parameters:
    cuda_tx: cuda_out
    host_tx: host_out
    cuda_tensor_pool: global/cuda_pools
    host_tensor_pool: global/host_pool
    stream_pool: global/stream_pool
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: cuda_out
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: host_out
    min_size: 1
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: 50
---
name: cuda_dotproduct_{i}
components:
- name: rx
  type: nvidia::gxf::DoubleBufferReceiver
  parameters:
    capacity: 2
- name: tx
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: tx
    min_size: 1
- type: nvidia::gxf::test::cuda::CublasDotProduct
  parameters:
    rx: rx
    tx: tx
    tensor_pool: global/cuda_dot_pool
---
name: host_dotproduct_{i}
components:
- name: rx
  type: nvidia::gxf::DoubleBufferReceiver
  parameters:
    capacity: 2
- name: tx
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: tx
    min_size: 1
- type: nvidia::gxf::test::cuda::HostDotProduct
  parameters:
    rx: rx
    tx: tx
    tensor_pool: global/host_dot_pool
---
name: copy2host_{i}
components:
- name: rx
  type: nvidia::gxf::DoubleBufferReceiver
- name: tx
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: tx
    min_size: 1
- type: nvidia::gxf::test::cuda::MemCpy2Host
  parameters:
    rx: rx
    tx: tx
    tensor_pool: global/host_dot_pool
---
name: streamsync_{i}
components:
- name: rx
  type: nvidia::gxf::DoubleBufferReceiver
- name: tx
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: tx
    min_size: 1
- type: nvidia::gxf::CudaStreamSync
  parameters:
    rx: rx
    tx: tx
---
name: verify_equal_{i}
components:
- name: rx0
  type: nvidia::gxf::DoubleBufferReceiver
- name: rx1
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::test::cuda::VerifyEqual
  parameters:
    rx0: rx0
    rx1: rx1
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx0
    min_size: 1
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx1
    min_size: 1
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: 50
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: stream_tensor_generator_{i}/host_out
    target: host_dotproduct_{i}/rx
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: stream_tensor_generator_{i}/cuda_out
    target: cuda_dotproduct_{i}/rx
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: cuda_dotproduct_{i}/tx
    target: copy2host_{i}/rx
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: copy2host_{i}/tx
    target: streamsync_{i}/rx
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: streamsync_{i}/tx
    target: verify_equal_{i}/rx0
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: host_dotproduct_{i}/tx
    target: verify_equal_{i}/rx1
""".format(i = "c" + str(i))

def create_app(n):
  header = """# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
%YAML 1.2
name: global
components:
- name: stream_pool
  type: nvidia::gxf::CudaStreamPool
  parameters:
    dev_id: 0
    stream_flags: 0
    stream_priority: 0
    reserved_size: 1
    max_size: 1110
- name: cuda_pools
  type: nvidia::gxf::BlockMemoryPool
  parameters:
    storage_type: 1 # cuda
    block_size: 8388608
    num_blocks: 15
- name: host_pool
  type: nvidia::gxf::BlockMemoryPool
  parameters:
    storage_type: 0 # host
    block_size: 8388608
    num_blocks: 15
- name: cuda_dot_pool
  type: nvidia::gxf::BlockMemoryPool
  parameters:
    storage_type: 1 # cuda
    block_size: 16384
    num_blocks: 15
- name: host_dot_pool
  type: nvidia::gxf::BlockMemoryPool
  parameters:
    storage_type: 0 # host
    block_size: 16384
    num_blocks: 15
---
"""

  footer = """---
components:
- name: clock
  type: nvidia::gxf::ManualClock
- type: nvidia::gxf::GreedyScheduler
  parameters:
    clock: clock
    max_duration_ms: 1000
"""

  return header + "".join([create_rx_tx(i) for i in range(n)]) + footer

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_cuda_stream_dotproduct_gen.yaml', create_app(85))
