/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/tests/test_clock_sync_helpers.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x948b019805404f63, 0x05ccc7d409024feb, "TestClockSyncHelpers",
                         "Extension for testing  graph clock synchronization components",
                         "NVIDIA", "1.1.0", "NVIDIA");

GXF_EXT_FACTORY_ADD(0x50d4440f0fac47f4, 0x4da733735882429a,
                    nvidia::gxf::test::ClockSetter, nvidia::gxf::Codelet,
                    "Sets provided clock to user-provided timestamps on tick");

GXF_EXT_FACTORY_ADD(0x4f78dfd3337542da, 0x2f67e8871d574fee,
                    nvidia::gxf::test::ClockChecker, nvidia::gxf::Codelet,
                    "Checks if provided clock matches user-provided timestamps on tick");

GXF_EXT_FACTORY_END()
