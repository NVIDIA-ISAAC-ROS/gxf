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
import random

'''
This stress script generates yaml file which has root behavior tree entity
having 1000 nested child behavior tree entities.
'''

behavior_parameter_dict= {
                          "SequenceBehavior": "children: [child{k}/child{k}_st ]",
                          "SwitchBehavior": "children: [child{k}/child{k}_st]\n    desired_behavior: 0",
                          "ParallelBehavior": "success_threshold: -1\n    failure_threshold: 1\n    children: [child{k}/child{k}_st]",
                          "SelectorBehavior": "children: [child{k}/child{k}_st]",
                          }
leaf_behavior_parameter_dict= {
                               "ConstantBehavior": "constant_status: {j}",
                               "TimerBehavior": "clock: sched/clock\n    switch_status: {j}\n    delay: 2",
                              }

def create_child_entity(i, n):
  if(i < n-1):
    behavior = random.choice(list(behavior_parameter_dict))
  else:
    behavior = random.choice(list(leaf_behavior_parameter_dict))
  return """---
name: child{i}
components:
- name: child{i}_controller
  type: nvidia::gxf::EntityCountFailureRepeatController
  parameters:
    max_repeat_count: 0
- name: child{i}_st
  type: nvidia::gxf::BTSchedulingTerm
  parameters:
    is_root: false
- name: child{i}_codelet
  type: nvidia::gxf::{behavior}
  parameters:
    {parameters}
    s_term: child{i}_st
""".format(i = str(i),
     behavior = behavior,
     parameters = behavior_parameter_dict.get(behavior).format(k = str(i+1)) if i < n-1 else leaf_behavior_parameter_dict.get(behavior).format(j = 0)
     )

def create_app(n):
  header = """%YAML 1.2
---
name: root
components:
- name: root_controller
  type: nvidia::gxf::EntityCountFailureRepeatController
  parameters:
    max_repeat_count: 0
- name: root_st
  type: nvidia::gxf::BTSchedulingTerm
  parameters:
    is_root: true
- name: root_codelet
  type: nvidia::gxf::SelectorBehavior
  parameters:
    s_term: root_st
    children: [ child0/child0_st ]
"""

  footer = """---
name: sched
components:
- name: clock
  type: nvidia::gxf::RealtimeClock
- name: greedy_scheduler
  type: nvidia::gxf::GreedyScheduler
  parameters:
    max_duration_ms: 100000000
    clock: clock
- name: job_stats
  type: nvidia::gxf::JobStatistics
  parameters:
    clock: clock
"""

  return header +"".join([create_child_entity(i, n) for i in range(n)]) + footer

def save(filename, text):
  with open(filename,'w') as file:
    file.write(text)

if __name__ == "__main__":
  out_dir = sys.argv[1]
  save(out_dir + '/test_stress_behavior_tree_nested_gen.yaml', create_app(1022))