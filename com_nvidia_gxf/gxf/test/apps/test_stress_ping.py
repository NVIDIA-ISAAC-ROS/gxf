"""
 SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
This stress script generates yaml file which has 100 pingtx and pingrx
components. StepCount codelet is used to validate expected number of
count.
'''
def create_ping_node(i):
    return """---
name: tx_{i}
components:
- name: signal
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
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
name: rx_{i}
components:
- name: signal
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: signal
    min_size: 1
- type: nvidia::gxf::PingRx
  parameters:
    signal: signal
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: 100
---
name: connections_{i}
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tx_{i}/signal
    target: rx_{i}/signal
""".format(i = str(i))

def create_app(n):
  header = """%YAML 1.2
"""

  footer = """---
name: scheduler
components:
- type: nvidia::gxf::MultiThreadScheduler
  parameters:
    max_duration_ms: 10000
    clock: clock
    worker_thread_number: 16
    stop_on_deadlock: false
- name: clock
  type: nvidia::gxf::RealtimeClock
- type: nvidia::gxf::JobStatistics
  parameters:
    clock: clock
"""

  return header + "".join([create_ping_node(i) for i in range(n)]) + footer

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_ping_gen.yaml', create_app(100))
