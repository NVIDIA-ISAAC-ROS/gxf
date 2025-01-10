/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/cuda/cuda_buffer.hpp"
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/cuda/cuda_scheduling_terms.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/cuda/cuda_stream_sync.hpp"
#include "gxf/cuda/stream_ordered_allocator.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xd63a98fa788211eb, 0xa917b38f664f399c, "CudaExtension",
                         "Cuda related components in Gxf Core", "Nvidia_Gxf", "2.6.0", "NVIDIA");

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

GXF_EXT_FACTORY_ADD(0x7397a4aaa4e94b6c, 0x850664d7c18c17ae,
                    nvidia::gxf::CudaStreamSchedulingTerm, nvidia::gxf::SchedulingTerm,
                    "A component which specifies that data is ready to be consumed "
                    "on a cuda stream based on callback function");

GXF_EXT_FACTORY_ADD(0xc2122c2d07454260, 0xb110514c3f4e82f9,
                    nvidia::gxf::CudaEventSchedulingTerm, nvidia::gxf::SchedulingTerm,
                    "A component which specifies that data is ready to be consumed "
                    "on a cuda stream based on event polling");

GXF_EXT_FACTORY_ADD_0(0x3b1098305f1e11ef, 0x8393c336b6b55546, nvidia::gxf::CudaBuffer,
                      "A container single block of cuda memory with support for async "
                      "memory allocations");

GXF_EXT_FACTORY_ADD(0x04f22de65f1d11ef, 0x9c8a373b588897af, nvidia::gxf::CudaAllocator,
                    nvidia::gxf::Allocator, "Async memory allocator interface based on cuda");

GXF_EXT_FACTORY_ADD(0x63d1d16813d711ef, 0x931a0be4a6378384, nvidia::gxf::StreamOrderedAllocator,
                    nvidia::gxf::CudaAllocator, "Allocator based on Cuda Stream Ordered"
                    " Memory Pools");

GXF_EXT_FACTORY_ADD(0xedab61b65f7611ef, 0xba73cfbc18c46142,
                    nvidia::gxf::CudaBufferAvailableSchedulingTerm,
                    nvidia::gxf::SchedulingTerm, "A component which specifies that data is "
                    "ready to be consumed in a cuda buffer based on callback function");


GXF_EXT_FACTORY_END()
