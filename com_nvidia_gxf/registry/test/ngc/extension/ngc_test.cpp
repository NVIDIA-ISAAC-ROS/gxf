/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0xfb5bcf1043f344bc, 0x846fd95a0a077045, "NGCTestExtension",
                           "Extension to test ngc features",
                           "NVIDIA", "1.4.0", "NVIDIA");
GXF_EXT_FACTORY_END()
