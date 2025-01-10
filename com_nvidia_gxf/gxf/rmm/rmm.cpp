/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/cuda/cuda_allocator.hpp"
#include "gxf/rmm/rmm_allocator.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x45fa20021f8311ef, 0xa380ef5f83754f84, "RMMExtension",
                         "RMM related components in Gxf Core", "Nvidia_Gxf",
                         "0.0.1", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("RMM Extension", "RMM", "GXF RMM Extension");

GXF_EXT_FACTORY_ADD(0x5a8ac4c21f8311ef, 0x8859d710d3299dfa,
                    nvidia::gxf::RMMAllocator, nvidia::gxf::CudaAllocator,
                    "Allocator based on RMM Memory Pools");

GXF_EXT_FACTORY_END()
