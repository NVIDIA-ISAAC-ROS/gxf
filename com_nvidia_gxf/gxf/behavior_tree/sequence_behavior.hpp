/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVIDIA_GXF_BT_SEQUENCE_BEHAVIOR_HPP_
#define NVIDIA_GXF_BT_SEQUENCE_BEHAVIOR_HPP_

#include <vector>

#include "common/fixed_vector.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/controller.hpp"
#include "gxf/std/scheduling_terms.hpp"

namespace nvidia {
namespace gxf {

// Sequence Behavior
// Runs its child entities in sequence, in the order in which they are defined.
// Succeeds when all child entities succeed, or fails as soon as one child
// entity fails.
class SequenceBehavior : public Codelet {
 public:
  virtual ~SequenceBehavior() = default;

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t tick() override;

 private:
  // parent codelet points to children's scheduling terms to schedule child
  // entities
  Parameter<FixedVector<Handle<nvidia::gxf::BTSchedulingTerm>, kMaxComponents> >
      children_;
  FixedVector<Handle<nvidia::gxf::BTSchedulingTerm>, kMaxComponents> children;
  std::vector<gxf_uid_t> children_eid;
  // its own scheduling term to start/stop itself
  Parameter<Handle<nvidia::gxf::BTSchedulingTerm> > s_term_;
  Handle<nvidia::gxf::BTSchedulingTerm> s_term;
  size_t getNumChildren() const;
  entity_state_t GetChildStatus(size_t child_id);
  gxf_result_t startChild(size_t child_id);
  size_t current_child_id;
  SchedulingConditionType ready_conditions;
  SchedulingConditionType never_conditions;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_BT_SEQUENCE_BEHAVIOR_HPP_
