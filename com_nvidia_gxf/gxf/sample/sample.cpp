/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/sample/hello_world.hpp"
#include "gxf/sample/multi_ping_rx.hpp"
#include "gxf/sample/ping_batch_rx.hpp"
#include "gxf/sample/ping_rx.hpp"
#include "gxf/sample/ping_tx.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(
    0xa6ad78b6168211ec, 0x96210242ac130002, "SampleExtension",
    "Sample extension to demonstrate the use of GXF features", "NVIDIA",
    "1.3.0", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Sample Extension", "Sample", "GXF Sample Extension");

GXF_EXT_FACTORY_ADD(0x568ba758168811ec, 0x96210242ac130002, nvidia::gxf::PingTx,
                    nvidia::gxf::Codelet, "Sends an empty message entity via a Transmitter");
GXF_EXT_FACTORY_ADD(0x75acc5caeaaa11ed, 0xa05b0242ac120003, nvidia::gxf::PingBatchRx,
                    nvidia::gxf::Codelet,
                    "Receives a message entity from a Receiver of specified batch size");
GXF_EXT_FACTORY_ADD(0xa7fc439a168811ec, 0x96210242ac130002, nvidia::gxf::PingRx,
                    nvidia::gxf::Codelet, "Receives a message entity from a Receiver");
GXF_EXT_FACTORY_ADD(0xd7a53484a1a711ed, 0xb686138d2bc637ee, nvidia::gxf::MultiPingRx,
                    nvidia::gxf::Codelet, "Receives message entities from multiple receivers");
GXF_EXT_FACTORY_ADD(0x3d1e58fc74fa11ed, 0xa7eef7e728cd4c11, nvidia::gxf::HelloWorld,
                    nvidia::gxf::Codelet, "Prints a 'Hello world' string on execution");

GXF_EXT_FACTORY_END()
