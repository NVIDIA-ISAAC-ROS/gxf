# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

gxf_add_gtests(
    EXT_NAME BehaviorTree
    SOURCES
        test_behavior.cpp
    DEPENDS
        GTest::gtest_main
        GXF::core
        GXF::cuda
        GXF::std
        yaml-cpp::yaml-cpp
    BUILD_DEPENDS
        GXF::behavior_tree
        GXF::serialization
        gxf_core_manifest
    DATA_FILES
        test_behavior_tree_constant.yaml
        test_behavior_tree_epoch_sch_with_all_behaviors.yaml
        test_behavior_tree_epoch.yaml
        test_behavior_tree_multi_thread_scheduler.yaml
        test_behavior_tree_parallel_failure.yaml
        test_behavior_tree_parallel_success.yaml
        test_behavior_tree_parallel.yaml
        test_behavior_tree_sel_failure.yaml
        test_behavior_tree_sel.yaml
        test_behavior_tree_seq_failure.yaml
        test_behavior_tree_seq_repeat_controller.yaml
        test_behavior_tree_seq.yaml
        test_behavior_tree_switch.yaml
        test_behavior_tree_timer.yaml
        test_behavior_tree_with_non_behavior_tree_codelet_as_child.yaml
        test_stress_behavior_tree.yaml
)
