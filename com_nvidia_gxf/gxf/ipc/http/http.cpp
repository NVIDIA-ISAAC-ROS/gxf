/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/ipc/http/http_client.hpp"
#include "gxf/ipc/http/http_client_cpprest_impl.hpp"
#include "gxf/ipc/http/http_ipc_client.hpp"
#include "gxf/ipc/http/http_server.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x22a21c97f20b4c03 , 0xb2776e6f52303328, "HttpExtension",
                         "Http protocol related components in Gxf ", "Nvidia_Gxf", "0.5.0",
                         "LICENSE");

    GXF_EXT_FACTORY_ADD(0x700895468a59442e, 0xb91e7708f2640fca,
                       nvidia::gxf::HttpServer, nvidia::gxf::IPCServer,
                       "A light-weight http API server");
    GXF_EXT_FACTORY_ADD(0x695479d528db4a4e, 0xbbbda2b1af390297,
                       nvidia::gxf::HttpIPCClient, nvidia::gxf::IPCClient,
                       "A light-weight http API client");
    GXF_EXT_FACTORY_ADD(0xfb4e20e415c84cb4, 0xad879c161d326748,
                      nvidia::gxf::HttpClient, nvidia::gxf::Component,
                      "Interface for basic http client that works with http server");
    GXF_EXT_FACTORY_ADD(0x562dae415e704495, 0xa7e8d196cedf3f9f,
                       nvidia::gxf::CppRestHttpClient, nvidia::gxf::HttpClient,
                       "A light-weight http client implementation");

GXF_EXT_FACTORY_END()
