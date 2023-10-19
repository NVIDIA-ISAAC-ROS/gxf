/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <memory>
#include <utility>

#include "gxf/npp/nppi_mul_c.hpp"
#include "gxf/npp/nppi_set.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x98839b87ddb54e77, 0xb0df44c1a5ad136e, "NppExtension",
                         "Image processing components based on Nvidia Performance Primitives",
                         "NVIDIA", "2.3.0", "NVIDIA");
GXF_EXT_FACTORY_SET_DISPLAY_INFO("NPP Extension", "NPP", "GXF NPP Extension");
GXF_EXT_FACTORY_ADD(0xb5ab03a9f3d54525, 0xbc0cd48b60bac2a5, nvidia::gxf::NppiSet,
                    nvidia::gxf::Codelet, "Creates a tensor with constant values");
GXF_EXT_FACTORY_ADD(0xb2e2181547c44aee, 0x99bf9f3ef6665931, nvidia::gxf::NppiMulC,
                    nvidia::gxf::Codelet, "Multiplies a tensor with a constant factor");
GXF_EXT_FACTORY_END()
