/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/stream/stream_nvscisync.hpp"
#include "gxf/stream/stream_sync_id.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x918e6ad78e1a43aa, 0x9b49251d4b6072b0, "StreamExtension",
                         "Stream related components in Gxf ", "Nvidia_Gxf", "0.5.0", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Stream Extension", "Stream", "GXF Stream Extension");

GXF_EXT_FACTORY_ADD_0(0x65bda2a27cfd4dfe, 0xa80a53ad72f5c2a7, nvidia::gxf::StreamSyncId,
                      "Provides access to deduce stream sync handle.");

GXF_EXT_FACTORY_ADD(0x0011bee75d5343ee, 0xaafa61485a436bc4, nvidia::gxf::StreamSync,
                    nvidia::gxf::Component, "Provides access to GXF stream Sync.");

GXF_EXT_FACTORY_END()
