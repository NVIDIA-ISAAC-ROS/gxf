/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string>
#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"
#include "ucx_component_serializer.hpp"
#include "ucx_context.hpp"
#include "ucx_entity_serializer.hpp"
#include "ucx_receiver.hpp"
#include "ucx_serialization_buffer.hpp"
#include "ucx_transmitter.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x525f8a1adfb5426b, 0x8ddb00c3ac839994, "UcxExtension",
                        "Extension for Unified Communication X framework",
                        "NVIDIA", "0.8.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x1d9fcaf71db14992, 0x93ec714979f7d78d, nvidia::gxf::UcxSerializationBuffer,
                    nvidia::gxf::Endpoint,
                    "Serializer Buffer for UCX.");

GXF_EXT_FACTORY_ADD(0x6499430542604f5c, 0xac5f69da6dd6cfa5, nvidia::gxf::UcxComponentSerializer,
                    nvidia::gxf::ComponentSerializer,
                    "Component Serializer for UCX.");

GXF_EXT_FACTORY_ADD(0x14997aa44a014cd4, 0x86ab687f85a13f10, nvidia::gxf::UcxEntitySerializer,
                    nvidia::gxf::EntitySerializer,
                    "Entity Serializer for UCX.");

GXF_EXT_FACTORY_ADD(0xe961132b45d548b8, 0xac5d2bb1a4a42279, nvidia::gxf::UcxReceiver,
                    nvidia::gxf::Receiver,
                    "Component to receive UCX message.");

GXF_EXT_FACTORY_ADD(0x58165d0378b74696, 0xb20071621f90aee7, nvidia::gxf::UcxTransmitter,
                    nvidia::gxf::Transmitter,
                    "Component to send UCX message.");

GXF_EXT_FACTORY_ADD(0x755d20a5d794467d, 0xa86c290eb2c32052, nvidia::gxf::UcxContext,
                    nvidia::gxf::NetworkContext,
                    "Component to hold Ucx Context.");

GXF_EXT_FACTORY_END()

}  // extern "C"
