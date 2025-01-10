/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "grpc_client.hpp"
#include "grpc_server.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x62e7335cc55547c9, 0xa2d221991b7fd250, "GrpcExtension",
                         "Extension for GRPC based communication tools",
                         "Nvidia_Gxf", "0.5.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0xe6b2f6c057a8431, 0x925bfa476c3265b6,
                    nvidia::gxf::GrpcServer, nvidia::gxf::IPCServer,
                    "IPC Server implementation based on Grpc");
GXF_EXT_FACTORY_ADD(0xf05f8a46b4ce4e3d, 0xbd9c326680abea03,
                    nvidia::gxf::GrpcClient, nvidia::gxf::IPCClient,
                    "IPC Server implementation based on Grpc");

GXF_EXT_FACTORY_END()
