/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/behavior_tree/constant_behavior.hpp"
#include "gxf/behavior_tree/entity_count_failure_repeat_controller.hpp"
#include "gxf/behavior_tree/parallel_behavior.hpp"
#include "gxf/behavior_tree/repeat_behavior.hpp"
#include "gxf/behavior_tree/selector_behavior.hpp"
#include "gxf/behavior_tree/sequence_behavior.hpp"
#include "gxf/behavior_tree/switch_behavior.hpp"
#include "gxf/behavior_tree/timer_behavior.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x9e6e55d41bc911ec, 0x96210242ac130002,
                         "BehaviorTreeExtension",
                         "Extension with behavior tree components",
                         "Nvidia_Gxf", "0.6.0", "NVIDIA");
GXF_EXT_FACTORY_SET_DISPLAY_INFO("Behavior Tree Extension", "Behavior Tree",
                                 "GXF Behavior Tree Extension");
GXF_EXT_FACTORY_ADD(
    0x1f68ff76e5b8414f, 0xab5c8cef935d81c7,
    nvidia::gxf::EntityCountFailureRepeatController, nvidia::gxf::Controller,
    "Restart controller that only terminates an entity after restarting");
GXF_EXT_FACTORY_ADD(0x11ae05e575a54246, 0x8a04ef37d45419ab,
                    nvidia::gxf::ConstantBehavior, nvidia::gxf::Codelet,
                    "Constant Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_ADD(0x1f5efe9768a54a91, 0x899037c0a222646f,
                    nvidia::gxf::SelectorBehavior, nvidia::gxf::Codelet,
                    "Selector Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_ADD(0x9e610c81e3074654, 0x8d73c3405050cea7,
                    nvidia::gxf::SequenceBehavior, nvidia::gxf::Codelet,
                    "Sequence Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_ADD(0xfbaafbf2644c4f49, 0xa398463215f54474,
                    nvidia::gxf::SwitchBehavior, nvidia::gxf::Codelet,
                    "Switch Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_ADD(0x9d035af9a33e447c, 0xa6ddc67b1d8ce687,
                    nvidia::gxf::RepeatBehavior, nvidia::gxf::Codelet,
                    "Repeat Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_ADD(0xdc48e75e5c7648a1, 0xb22bc2b81d83a7b6,
                    nvidia::gxf::ParallelBehavior, nvidia::gxf::Codelet,
                    "Parallel Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_ADD(0x23cc558577084b93, 0x83798b46c3ab56d5,
                    nvidia::gxf::TimerBehavior, nvidia::gxf::Codelet,
                    "Timer Behavior Codelet of Behavior Tree");
GXF_EXT_FACTORY_END()
