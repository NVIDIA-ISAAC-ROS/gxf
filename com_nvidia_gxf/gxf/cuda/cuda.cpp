/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/cuda/cuda_stream_sync.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xd63a98fa788211eb, 0xa917b38f664f399c, "CudaExtension",
                         "Cuda related components in Gxf Core", "Nvidia_Gxf", "2.3.0", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Cuda Extension", "Cuda", "GXF Cuda Extension");

GXF_EXT_FACTORY_ADD_0(0x5683d692788411eb, 0x9338c3be62d576be, nvidia::gxf::CudaStream,
                      "Provides access to cuda stream handle.");

GXF_EXT_FACTORY_ADD_0(0x7982aeac37f141be, 0xade86f00b4b5d47c, nvidia::gxf::CudaStreamId,
                      "Provides access to deduce cuda stream handle.");

GXF_EXT_FACTORY_ADD_0(0xf5388d5ca70947e7, 0x86c4171779bc64f3, nvidia::gxf::CudaEvent,
                      "Provides access to cuda event.");

GXF_EXT_FACTORY_ADD(0x6733bf8bba5e4fae, 0xb596af2d1269d0e7, nvidia::gxf::CudaStreamPool,
                    nvidia::gxf::Allocator, "A Cuda stream pool provides stream allocation");

GXF_EXT_FACTORY_ADD(0x0d1d81426648485d, 0x97d5277eed00129c, nvidia::gxf::CudaStreamSync,
                    nvidia::gxf::Codelet, "Synchronize all cuda streams which are "
                    "carried by message entities");

GXF_EXT_FACTORY_END()
