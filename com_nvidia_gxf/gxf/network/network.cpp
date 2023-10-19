/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/network/clock_sync_primary.hpp"
#include "gxf/network/clock_sync_secondary.hpp"
#include "gxf/network/tcp_client.hpp"
#include "gxf/network/tcp_codelet.hpp"
#include "gxf/network/tcp_server.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xf50665e5ade2f71b, 0xde2a2380614b1725, "NetworkExtension",
                         "Extension for communications external to a computation graph",
                         "Nvidia_Gxf", "1.3.0", "NVIDIA");
GXF_EXT_FACTORY_SET_DISPLAY_INFO("Network Extension", "Network", "GXF Network Extension");

GXF_EXT_FACTORY_ADD(0xa61832d1b0f942b3, 0x97b2ccec0e864e61, nvidia::gxf::ClockSyncPrimary,
                    nvidia::gxf::Codelet,
                    "Publishes application clock timestamp for use by other apps");

GXF_EXT_FACTORY_ADD(0xe84945fa86304516, 0xf7ce7df2b05947c7, nvidia::gxf::ClockSyncSecondary,
                    nvidia::gxf::Codelet,
                    "Advances application SyntheticClock to received timestamp");

GXF_EXT_FACTORY_ADD(0x620c572cf03d11ed, 0x8a228f9adbb0e784, nvidia::gxf::TcpCodelet,
                    nvidia::gxf::Codelet,
                    "Interface for a codelet for either end of a TCP connection");

GXF_EXT_FACTORY_ADD(0x956fa19e58dd7f15, 0xdf000d2d6eaf8f70, nvidia::gxf::TcpClient,
                    nvidia::gxf::TcpCodelet,
                    "Codelet that functions as a client in a TCP connection");

GXF_EXT_FACTORY_ADD(0xa3e0e42de32e73ab, 0xef83fbb311310759, nvidia::gxf::TcpServer,
                    nvidia::gxf::TcpCodelet,
                    "Codelet that functions as a server in a TCP connection");

GXF_EXT_FACTORY_END()
