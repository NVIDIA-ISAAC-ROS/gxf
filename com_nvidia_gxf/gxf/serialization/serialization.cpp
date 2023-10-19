/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/serialization/component_serializer.hpp"
#include "gxf/serialization/endpoint.hpp"
#include "gxf/serialization/entity_recorder.hpp"
#include "gxf/serialization/entity_replayer.hpp"
#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/serialization/file.hpp"
#include "gxf/serialization/serialization_buffer.hpp"
#include "gxf/serialization/std_component_serializer.hpp"
#include "gxf/serialization/std_entity_id_serializer.hpp"
#include "gxf/serialization/std_entity_serializer.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xbc573c2f89b3d4b0, 0x80612da8b11fe79a, "SerializationExtension",
                         "Extension for serializing messages",
                         "NVIDIA", "2.3.0", "NVIDIA");

GXF_EXT_FACTORY_SET_DISPLAY_INFO("Serialization Extension", "Serialization",
                                 "GXF Serialization Extension");
// -------------------------------------------------------------------------------------------------
// -- Endpoints ------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
GXF_EXT_FACTORY_ADD(0x801449dd1657513f, 0xa3f8f648047583d0,
                    nvidia::gxf::Endpoint, nvidia::gxf::Component,
                    "Interface for exchanging data external to an application graph");
GXF_EXT_FACTORY_ADD(0xcb253dcecce62ff4, 0x84ef3cf1db6e24a3,
                    nvidia::gxf::SerializationBuffer, nvidia::gxf::Endpoint,
                    "Buffer to hold serialized data");
GXF_EXT_FACTORY_ADD(0xc132f22c1ed16c0f, 0xed53c63bf9b28354,
                    nvidia::gxf::File, nvidia::gxf::Endpoint,
                    "Wrapper around C file I/O API");

// -------------------------------------------------------------------------------------------------
// -- Component Serializers ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
GXF_EXT_FACTORY_ADD(0x8c76a82821771484, 0xf841d39c3fa47613,
                    nvidia::gxf::ComponentSerializer, nvidia::gxf::Component,
                    "Interface for serializing components");
GXF_EXT_FACTORY_ADD(0xc0e6b36c39ac50ac, 0xce8d702e18d8bff7,
                    nvidia::gxf::StdComponentSerializer, nvidia::gxf::ComponentSerializer,
                    "Serializer for Timestamp and Tensor components");

// -------------------------------------------------------------------------------------------------
// -- Entity Serializers ---------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
GXF_EXT_FACTORY_ADD(0xfd7892f70464e591, 0xdef388d77c8dde70,
                    nvidia::gxf::EntitySerializer, nvidia::gxf::Component,
                    "Interface for serializing entities");
GXF_EXT_FACTORY_ADD(0xbe82d2fa52f53687, 0x84d3e4999125a09b,
                    nvidia::gxf::StdEntitySerializer, nvidia::gxf::EntitySerializer,
                    "Serializes entities for sharing data between GXF applications");

GXF_EXT_FACTORY_ADD(0x0d0fa475c5324a45, 0x890028acd3adf8c7,
                    nvidia::gxf::StdEntityIdSerializer, nvidia::gxf::EntitySerializer,
                    "Serializes entity ID for sharing between GXF applications");

// -------------------------------------------------------------------------------------------------
// -- Codelets -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
GXF_EXT_FACTORY_ADD(0x9d5955c78fda22c7, 0xf18fea5e2d195be9,
                    nvidia::gxf::EntityRecorder, nvidia::gxf::Codelet,
                    "Serializes incoming messages and writes them to a file");
GXF_EXT_FACTORY_ADD(0xfe827c12d360c63c, 0x809432b9244d83b6,
                    nvidia::gxf::EntityReplayer, nvidia::gxf::Codelet,
                    "Deserializes and publishes messages from a file");

GXF_EXT_FACTORY_END()
