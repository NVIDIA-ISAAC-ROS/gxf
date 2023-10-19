/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(kNullUid, 0x1b12ffebc2504ced, "SampleTestExtension",
                         "Sample test extension to test API GxfRuntimeInfo", "NVIDIA", "0.1.0",
                         "NVIDIA");

GXF_EXT_FACTORY_END()
