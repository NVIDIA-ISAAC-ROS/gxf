/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/multimedia/audio.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0x6f2d1afc1057481a, 0x9da6a5f61fed178e, "MultimediaExtension",
                           "Multimedia related data types, interfaces and components in Gxf Core",
                           "NVIDIA", "2.3.0", "NVIDIA");
  GXF_EXT_FACTORY_SET_DISPLAY_INFO("Multimedia Extension", "Multimedia",
                                   "GXF Multimedia Extension");
  GXF_EXT_FACTORY_ADD_0(0xa914cac65f19449d, 0x9ade8c5cdcebe7c3,
                        nvidia::gxf::AudioBuffer,
                        "Holds information about a single audio frame");
  GXF_EXT_FACTORY_ADD_0(0x16ad58c8b463422c, 0xb09761a9acc5050e,
                        nvidia::gxf::VideoBuffer,
                        "Holds information about a single video frame");
  GXF_EXT_FACTORY_ADD_0(0xbac444bdf13b4927, 0x85c1db5e1947f890,
                        nvidia::gxf::CameraModel,
                        "Holds information of camera intrinsics");
  GXF_EXT_FACTORY_ADD_0(0xda8387d9d8ab425f, 0x909b814694f148ef,
                        nvidia::gxf::Pose3D,
                        "Holds information of camera extrinsics");
GXF_EXT_FACTORY_END()
