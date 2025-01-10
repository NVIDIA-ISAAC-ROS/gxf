"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
This stress script generates yaml file which has 100 graphs containing gather
subgraph. Each instance of gather subgraph has 2 Forward subgraphs.
'''
def create_graphs_containing_subgraph(i):
    return """---
name: tx{i}_0
components:
- name: signal
  type: nvidia::gxf::test::MockTransmitter
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: signal
    min_size: 1
- type: nvidia::gxf::PingTx
  parameters:
    signal: signal
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: 100
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: 100
---
name: tx{i}_1
components:
- name: signal
  type: nvidia::gxf::test::MockTransmitter
- type: {DownstreamReceptiveSchedulingTerm}
  parameters:
    transmitter: signal
    min_size: 1
- type: nvidia::gxf::PingTx
  parameters:
    signal: signal
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: 100
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: 100
---
name: rx{i}
components:
- name: signal
  type: nvidia::gxf::test::MockReceiver
  parameters:
    max_capacity: 2
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: signal
    min_size: 1
- type: nvidia::gxf::test::PingBatchRx
  parameters:
    signal: signal
    batch_size: 2
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: 100
---
name: gather_subgraph{i}
components:
- type: nvidia::gxf::Subgraph
  name: gather_subgraph{i}
  parameters:
    location: "gxf/test/apps/gather_subgraph.yaml"
    prerequisites:
      monitored_rx: rx{i}/signal
- name: output
  parameters:
    max_capacity: 2
- name: forward1_buf_term
  parameters:
    min_size: 1
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx{i}_0/signal
    target: gather_subgraph{i}/input1
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx{i}_1/signal
    target: gather_subgraph{i}/input2
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: gather_subgraph{i}/output
    target: rx{i}/signal
""".format(i = str(i),
           DownstreamReceptiveSchedulingTerm = "nvidia::gxf::DownstreamReceptiveSchedulingTerm")

def create_app(n):
  header = """%YAML 1.2
"""

  footer = """---
components:
- name: clock
  type: nvidia::gxf::RealtimeClock
- type: nvidia::gxf::GreedyScheduler
  parameters:
    max_duration_ms: 1000000
    clock: clock
- type: nvidia::gxf::test::EntityMonitor
- type: nvidia::gxf::JobStatistics
  parameters:
    clock: clock
"""

  return header + "".join([create_graphs_containing_subgraph(i) for i in range(n)]) + footer

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_subgraph_gen.yaml', create_app(30))
