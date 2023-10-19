/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <chrono>
#include <cstring>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/component.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/epoch_scheduler.hpp"

namespace {
constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";
constexpr const char* kExtensions[] = {
    "gxf/std/libgxf_std.so",
    "gxf/serialization/libgxf_serialization.so",
    "gxf/behavior_tree/libgxf_behavior_tree.so",
    "gxf/test/extensions/libgxf_test.so",
};
}  // namespace

TEST(Entity, BehaviorTreeSwitch) {
  // Test Switch Behavior
  gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_switch.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_scene, eid_ref;
  gxf_uid_t eid_pose, eid_det, eid_seg;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "scene", &eid_scene));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "ref", &eid_ref));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "pose", &eid_pose));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "det", &eid_det));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "seg", &eid_seg));
  entity_state_t root_status, scene_status, ref_status, pose_status,
      det_status, seg_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_scene, &scene_status));
  GXF_ASSERT_EQ(scene_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_ref, &ref_status));
  GXF_ASSERT_EQ(ref_status, GXF_BEHAVIOR_INIT);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_pose, &pose_status));
  GXF_ASSERT_EQ(pose_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_det, &det_status));
  GXF_ASSERT_EQ(det_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_seg, &seg_status));
  GXF_ASSERT_EQ(seg_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeSelector) {
  // Test Selector Behavior
  gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_sel.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_door_dist, eid_door_det, eid_knock_on_door;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "door_distance", &eid_door_dist));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "door_detected", &eid_door_det));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "knock", &eid_knock_on_door));
  entity_state_t root_status, door_dist_status, door_det_status,
      knock_on_door_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(
      GxfEntityGetState(context, eid_door_dist, &door_dist_status));
  GXF_ASSERT_EQ(door_dist_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(
      GxfEntityGetState(context, eid_door_det, &door_det_status));
  GXF_ASSERT_EQ(door_det_status, GXF_BEHAVIOR_INIT);
  GXF_ASSERT_SUCCESS(
      GxfEntityGetState(context, eid_knock_on_door, &knock_on_door_status));
  GXF_ASSERT_EQ(knock_on_door_status, GXF_BEHAVIOR_INIT);
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeSelectorFailure) {
  // Test Selector Behavior Failure cases
  gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0, nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(
      GxfGraphLoadFile(context, "gxf/behavior_tree/tests/test_behavior_tree_sel_failure.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_child1, eid_child2, eid_child3, eid_child4;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child3", &eid_child3));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child4", &eid_child4));
  entity_state_t root_status, child1_status, child2_status, child3_status, child4_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child3, &child3_status));
  GXF_ASSERT_EQ(child3_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child4, &child4_status));
  GXF_ASSERT_EQ(child4_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeSequence) {
  // Test Sequence Behavior
 gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_seq.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_scene, eid_ref;
  gxf_uid_t eid_pose, eid_det, eid_seg;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "scene", &eid_scene));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "ref", &eid_ref));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "pose", &eid_pose));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "det", &eid_det));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "seg", &eid_seg));
  entity_state_t root_status, scene_status, ref_status, pose_status,
      det_status, seg_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_scene, &scene_status));
  GXF_ASSERT_EQ(scene_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_ref, &ref_status));
  GXF_ASSERT_EQ(ref_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_pose, &pose_status));
  GXF_ASSERT_EQ(pose_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_det, &det_status));
  GXF_ASSERT_EQ(det_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_seg, &seg_status));
  GXF_ASSERT_EQ(seg_status, GXF_BEHAVIOR_SUCCESS);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeSequenceFailure) {
  // Test Sequence Behavior
 gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_seq_failure.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_scene, eid_ref;
  gxf_uid_t eid_pose, eid_det, eid_seg;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "scene", &eid_scene));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "ref", &eid_ref));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "pose", &eid_pose));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "det", &eid_det));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "seg", &eid_seg));
  entity_state_t root_status, scene_status, ref_status, pose_status,
      det_status, seg_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_scene, &scene_status));
  GXF_ASSERT_EQ(scene_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_ref, &ref_status));
  GXF_ASSERT_EQ(ref_status, GXF_BEHAVIOR_INIT);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_pose, &pose_status));
  GXF_ASSERT_EQ(pose_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_det, &det_status));
  GXF_ASSERT_EQ(det_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_seg, &seg_status));
  GXF_ASSERT_EQ(seg_status, GXF_BEHAVIOR_FAILURE);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeParallel) {
  // Test Parallel Behavior
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1,
                                   nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_parallel.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_child1, eid_child2;
  gxf_uid_t eid_door_distance, eid_door_detected;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  GXF_ASSERT_SUCCESS(
      GxfEntityFind(context, "door_distance", &eid_door_distance));
  GXF_ASSERT_SUCCESS(
      GxfEntityFind(context, "door_detected", &eid_door_detected));
  entity_state_t root_status, child1_status, child2_status,
      door_distance_status, door_detected_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(
      GxfEntityGetState(context, eid_door_distance, &door_distance_status));
  GXF_ASSERT_EQ(door_distance_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(
      GxfEntityGetState(context, eid_door_detected, &door_detected_status));
  GXF_ASSERT_EQ(door_detected_status, GXF_BEHAVIOR_SUCCESS);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeParallelSuccess) {
  // Test Parallel Behavior
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(
      GxfGraphLoadFile(context, "gxf/behavior_tree/tests/test_behavior_tree_parallel_success.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_child1, eid_child2, eid_child3, eid_child4, eid_child5, eid_child6,
      eid_child7;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child3", &eid_child3));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child4", &eid_child4));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child5", &eid_child5));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child6", &eid_child6));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child7", &eid_child7));

  entity_state_t root_status, child1_status, child2_status, child3_status, child4_status, child5_status, child6_status, child7_status ;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child3, &child3_status));
  GXF_ASSERT_EQ(child3_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child4, &child4_status));
  GXF_ASSERT_EQ(child4_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child5, &child5_status));
  GXF_ASSERT_EQ(child5_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child6, &child6_status));
  GXF_ASSERT_EQ(child6_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child7, &child7_status));
  GXF_ASSERT_EQ(child7_status, GXF_BEHAVIOR_FAILURE);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeParallelFailuer) {
  // Test Parallel Behavior
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(
      GxfGraphLoadFile(context, "gxf/behavior_tree/tests/test_behavior_tree_parallel_failure.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_child1, eid_child2, eid_child3, eid_child4, eid_child5, eid_child6,
      eid_child7;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child3", &eid_child3));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child4", &eid_child4));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child5", &eid_child5));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child6", &eid_child6));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child7", &eid_child7));

  entity_state_t root_status, child1_status, child2_status, child3_status, child4_status, child5_status, child6_status, child7_status ;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child3, &child3_status));
  GXF_ASSERT_EQ(child3_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child4, &child4_status));
  GXF_ASSERT_EQ(child4_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child5, &child5_status));
  GXF_ASSERT_EQ(child5_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child6, &child6_status));
  GXF_ASSERT_EQ(child6_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child7, &child7_status));
  GXF_ASSERT_EQ(child7_status, GXF_BEHAVIOR_FAILURE);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeTimer) {
  // Test Timer Behavior
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1,
                                   nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_timer.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_child1;
  gxf_uid_t eid_switch;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "switch", &eid_switch));
  entity_state_t root_status, child1_status, switch_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_switch, &switch_status));
  GXF_ASSERT_EQ(switch_status, GXF_BEHAVIOR_FAILURE);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeConstant) {
  // Test Constant Behavior
  gxf_context_t context;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  const GxfLoadExtensionsInfo info{nullptr, 0, &kGxeManifestFilename, 1,
                                   nullptr};
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &info));
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_constant.yaml"));
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  gxf_uid_t eid_root, eid_child1, eid_child2;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  entity_state_t root_status, child1_status, child2_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_FAILURE);

  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(Entity, BehaviorTreeEpoch) {
  // Test Constant Behavior
  gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_epoch.yaml"));


  gxf_uid_t eid = 0;
  gxf_uid_t cid = 0;
  int32_t offset = 0;
  gxf_tid_t tid{0, 0};
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "sched", &eid));
  GXF_ASSERT_SUCCESS(
      GxfComponentFind(context, eid, tid, "epoch", &offset, &cid));


  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));


 std::this_thread::sleep_for(std::chrono::milliseconds(5));

  nvidia::gxf::EpochScheduler* scheduler_ptr = nullptr;

  GXF_ASSERT_SUCCESS(GxfComponentPointer(
      context, cid, tid, reinterpret_cast<void**>(&scheduler_ptr)));

  EXPECT_NE(scheduler_ptr, nullptr);

  auto result = scheduler_ptr->runEpoch(50.0);

  EXPECT_TRUE(result);
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));
  gxf_uid_t eid_root, eid_child1, eid_child2;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  entity_state_t root_status, child1_status, child2_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_FAILURE);

  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}


TEST(Entity, BehaviorTreeEpochAllBehavior) {
  // Test All Behaviors
  gxf_context_t context;
  const GxfLoadExtensionsInfo load_extension_info{kExtensions, 4, nullptr, 0,
                                                  nullptr};
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  ASSERT_EQ(GxfLoadExtensions(context, &load_extension_info), GXF_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfGraphLoadFile(
      context, "gxf/behavior_tree/tests/test_behavior_tree_epoch_sch_with_all_behaviors.yaml"));

  gxf_uid_t eid = 0;
  gxf_uid_t cid = 0;
  int32_t offset = 0;
  gxf_tid_t tid{0, 0};
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "sched", &eid));
  GXF_ASSERT_SUCCESS(
      GxfComponentFind(context, eid, tid, "epoch", &offset, &cid));


  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));


 std::this_thread::sleep_for(std::chrono::milliseconds(5));

  nvidia::gxf::EpochScheduler* scheduler_ptr = nullptr;

  GXF_ASSERT_SUCCESS(GxfComponentPointer(
      context, cid, tid, reinterpret_cast<void**>(&scheduler_ptr)));

  EXPECT_NE(scheduler_ptr, nullptr);

  auto result = scheduler_ptr->runEpoch(20000.0);

  EXPECT_TRUE(result);
  GXF_ASSERT_SUCCESS(GxfGraphInterrupt(context));

  gxf_uid_t eid_root, eid_child1, eid_child2, eid_child3, eid_child4, eid_child5, eid_child6,
      eid_child7, eid_child8, eid_child9, eid_child10;
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "root", &eid_root));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child1", &eid_child1));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child2", &eid_child2));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child3", &eid_child3));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child4", &eid_child4));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child5", &eid_child5));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child6", &eid_child6));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child7", &eid_child7));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child8", &eid_child8));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child9", &eid_child9));
  GXF_ASSERT_SUCCESS(GxfEntityFind(context, "child10", &eid_child10));
  entity_state_t root_status, child1_status, child2_status, child3_status, child4_status,
      child5_status, child6_status, child7_status, child8_status, child9_status, child10_status;
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_root, &root_status));
  GXF_ASSERT_EQ(root_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child1, &child1_status));
  GXF_ASSERT_EQ(child1_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child2, &child2_status));
  GXF_ASSERT_EQ(child2_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child3, &child3_status));
  GXF_ASSERT_EQ(child3_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child4, &child4_status));
  GXF_ASSERT_EQ(child4_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child5, &child5_status));
  GXF_ASSERT_EQ(child5_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child6, &child6_status));
  GXF_ASSERT_EQ(child6_status, GXF_BEHAVIOR_INIT);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child7, &child7_status));
  GXF_ASSERT_EQ(child7_status, GXF_BEHAVIOR_SUCCESS);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child8, &child8_status));
  GXF_ASSERT_EQ(child8_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child9, &child9_status));
  GXF_ASSERT_EQ(child9_status, GXF_BEHAVIOR_FAILURE);
  GXF_ASSERT_SUCCESS(GxfEntityGetState(context, eid_child10, &child10_status));
  GXF_ASSERT_EQ(child10_status, GXF_BEHAVIOR_SUCCESS);

  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}
