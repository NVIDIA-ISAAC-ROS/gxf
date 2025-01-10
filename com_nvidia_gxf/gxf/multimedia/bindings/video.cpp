/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <iostream>

#include "gxf/core/gxf.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/eos.hpp"
#include "gxf/std/tensor.hpp"

namespace {

// Gets Python buffer format string from element type of the tensor
/**
 * For pybind11 dtype mapping, please refer the implementation of
 * template <typename T>
 *  struct format_descriptor<
 * https://github.com/pybind/pybind11/blob/master/include/pybind11/detail/common.h
*/
const char* PythonBufferFormatString(nvidia::gxf::PrimitiveType element_type) {
  switch (element_type) {
    case nvidia::gxf::PrimitiveType::kCustom:
      return "O";  // Use "O" for a generic Python object
    case nvidia::gxf::PrimitiveType::kUnsigned8:
      return "B";
    case nvidia::gxf::PrimitiveType::kUnsigned16:
      return "H";
    case nvidia::gxf::PrimitiveType::kUnsigned32:
      return "I";
    case nvidia::gxf::PrimitiveType::kUnsigned64:
      return "Q";
    case nvidia::gxf::PrimitiveType::kInt8:
      return "b";
    case nvidia::gxf::PrimitiveType::kInt16:
      return "h";
    case nvidia::gxf::PrimitiveType::kInt32:
      return "i";
    case nvidia::gxf::PrimitiveType::kInt64:
      return "q";
    case nvidia::gxf::PrimitiveType::kFloat16:
      return "e";
    case nvidia::gxf::PrimitiveType::kFloat32:
      return "f";
    case nvidia::gxf::PrimitiveType::kFloat64:
      return "d";
    case nvidia::gxf::PrimitiveType::kComplex64:
      return "Zf";
    case nvidia::gxf::PrimitiveType::kComplex128:
      return "Zd";
  }
  return "O";  // Use "O" for a generic Python object
}

}  // namespace

PYBIND11_MODULE(video_pybind, m) {
  pybind11::class_<nvidia::gxf::VideoBuffer>(m, "VideoBuffer", pybind11::buffer_protocol())
      /**
       * Implement the Python buffer protocol for the VideoBuffer class.
       * This allows the C++ VideoBuffer object to be exposed as a Python buffer object,
       * enabling efficient data sharing between C++ and Python without unnecessary copying.
       *
       * However, please be aware of the gap
       * 1. Python standard pybind11::buffer_info is designed for planar buffer
       * 2. GXF VideoBuffer does support multi-plane buffer
       */
      .def_buffer([](nvidia::gxf::VideoBuffer* vb) -> pybind11::buffer_info {
        std::vector<int> dimensions;
        std::vector<int> strides;
        // uint64_t bytes_per_pixel = 0;
        auto buffer_info = vb->video_frame_info();
        nvidia::gxf::Expected<nvidia::gxf::PrimitiveType> primitive_type =
            nvidia::gxf::VideoBuffer::getPlanarPrimitiveType(buffer_info.color_format);
        if (!primitive_type) { throw pybind11::value_error(GxfResultStr(primitive_type.error())); }

        const char* format = PythonBufferFormatString(primitive_type.value());

        auto c = static_cast<int32_t>(buffer_info.color_planes.size());
        auto h = static_cast<int32_t>(buffer_info.height);
        auto w = static_cast<int32_t>(buffer_info.width);

        if ((c < 1) || (h < 1) || (w < 1)) {
          GXF_LOG_ERROR(
              "VideoBuffer cannot be converted to tensor."
              " Invalid dimensions [CHW]:[%d,%d,%d]",
              c, h, w);
          throw pybind11::value_error(GxfResultStr(GXF_INVALID_DATA_FORMAT));
        }

        nvidia::gxf::Shape vb_shape;
        if (buffer_info.color_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB) {
          vb_shape = nvidia::gxf::Shape({w, h, 3});
        } else if (buffer_info.color_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
          vb_shape = nvidia::gxf::Shape({w, h, 4});
        }
        uint64_t curr = 1;
        for (size_t i = 0; i < vb_shape.rank(); i++) {
          curr = curr * vb_shape.dimension(i);
          strides.push_back(int(buffer_info.color_planes[0].size / curr));
        }
        for (size_t i = 0; i < vb_shape.rank(); i++) {
          dimensions.push_back(vb_shape.dimension(i));
        }
        return pybind11::buffer_info(vb->pointer(), /* Pointer to buffer */
                                     uint64_t(buffer_info.color_planes[0].bytes_per_pixel /
                                              vb_shape.rank()), /* Size of one scalar */
                                     format,          /* Python struct-style format descriptor */
                                     vb_shape.rank(), /* Number of dimensions */
                                     dimensions,      /* Buffer dimensions */
                                     strides          /* Stride dimensions */
        );
      })
      .def(pybind11::init([]() { return nvidia::gxf::VideoBuffer(); }),
           pybind11::return_value_policy::reference)
      /**
       * This Python VideoBuffer constructor only creates planar buffer to be consistent
       * with the buffer protocol defined above using pybind11::buffer_info,
       * specifically either RGB or RGBA.
       *
       * C++ VideoBuffer API has no such constraint, which can be used to create
       * multi-plane buffer.
      */
      .def(pybind11::init([](pybind11::object _image, nvidia::gxf::Allocator* allocator,
                             nvidia::gxf::MemoryStorageType type) {
             auto vb = nvidia::gxf::VideoBuffer();
             nvidia::gxf::VideoBufferInfo info{};
             void* data = NULL;
             const auto _type = std::string(pybind11::str(pybind11::type::of(_image)));
             // try {
             {
               if (_type.find("PIL.") != std::string::npos) {
                 _image.attr("load")();
                 const auto mode = _image.attr("mode").cast<std::string>();
                 if (mode != "RGB" && mode != "RGBA") {
                   pybind11::value_error("Only RGB and RGBA supported!");
                 }
                 info.height = _image.attr("height").cast<int>();
                 info.width = _image.attr("width").cast<int>();
                 info.surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
                 if (mode == "RGB") {
                   info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB;
                   nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>
                       color_format_size;
                   info.color_planes = color_format_size.getDefaultColorPlanes(
                       info.width, info.height,
                       false);  // VB[w, h, bytes_pixes]dtype =24  PIL [w, h, channels]dtype =8
                 } else {
                   info.color_format = nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
                   nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>
                       color_format_size;
                   info.color_planes =
                       color_format_size.getDefaultColorPlanes(info.width, info.height, false);
                 }
                 pybind11::buffer buffer = _image.attr("__array__")();
                 pybind11::buffer_info buffer_info = buffer.request();
                 data = buffer_info.ptr;
                 nvidia::gxf::MemoryBuffer::release_function_t release_func = [](void* pointer) {
                   GXF_LOG_DEBUG("Video Buffer object deleted. No memory released");
                   return nvidia::gxf::Success;
                 };
                 auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                     allocator->context(), allocator->cid());
                 if (!allocator_handle) {
                   throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
                 }
                 auto resize_result = vb.resizeCustom(info, static_cast<uint64_t>(buffer_info.size),
                                                      type, allocator_handle.value());
                 if (!resize_result) {
                   throw pybind11::value_error(GxfResultStr(resize_result.error()));
                 }
                 if (type == nvidia::gxf::MemoryStorageType::kHost ||
                     type == nvidia::gxf::MemoryStorageType::kSystem)
                   std::memcpy(vb.pointer(), data, info.color_planes[0].size);
                 else if (type == nvidia::gxf::MemoryStorageType::kDevice)
                   cudaMemcpy(vb.pointer(), data, info.color_planes[0].size,
                              cudaMemcpyHostToDevice);
               }
             }
             return vb;
           }),
           pybind11::return_value_policy::reference)
      /**
       * Please note it returns this VideoBuffer binding object reference,
       * not the mapped pybind11::buffer_info representation
       *
       * This is critical because multi-plane buffer will not get mis-interpreted,
       * and we can use our own get_info() method to interpret multi-plane buffer
      */
      .def_static(
          "get_from_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto maybe_video_buffer = e.get<nvidia::gxf::VideoBuffer>(name);
            if (!maybe_video_buffer) {
              auto eos = e.get<nvidia::gxf::EndOfStream>();
              if (!eos) {
                GXF_LOG_ERROR("Error getting video buffer called %s. And no EOS in the message", name);
                throw pybind11::value_error("error getting video buffer.");
              } else {
                GXF_LOG_INFO("No VideoBuffer but find EOS with stream id[%ld]", eos.value()->stream_id());
                throw pybind11::value_error("no video buffer and EOS received");
              }
            }
            auto vb = maybe_video_buffer.value().get();
            return vb;
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto result = e.add<nvidia::gxf::VideoBuffer>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            return result.value().get();
          },
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, nvidia::gxf::VideoBuffer& v, const char* name = nullptr) {
            auto result = e.add<nvidia::gxf::VideoBuffer>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            auto new_video_buffer = result.value().get();
            *new_video_buffer = std::move(v);
            return;
          },
          "Add a video_buffer to entity", pybind11::arg("message"), pybind11::arg("video_buffer"),
          pybind11::arg("name") = "", pybind11::return_value_policy::reference)
      /**
       * Please note this get_info is aligned with pybind11::buffer_info.
       * Other info from GXF VideoBuffer like color format or offsets will be
       * exposed by other APIs
      */
      .def(
          "get_info",
          [](nvidia::gxf::VideoBuffer& v) {
            std::vector<uint32_t> dimensions;
            std::vector<uint32_t> strides;
            std::vector<uint32_t> offsets;
            std::string color_format;

            auto buffer_info = v.video_frame_info();
            auto h = buffer_info.height;
            auto w = buffer_info.width;

            // Get buffer per element data type
            auto result = v.getPlanarPrimitiveType(buffer_info.color_format);
            if (!result) { pybind11::value_error("Error getting primitive type of video buffer"); }
            auto descriptor = result.value();

            // Get buffer dimensional info
            switch (buffer_info.color_format) {
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
                dimensions = {h, w, 3};
                strides = {w * 3, 3, 1};
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA: {
                dimensions = {h, w, 4};
                strides = {w * 4, 4, 1};
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12: {
                // For NV12 from GStreamer
                dimensions = {2, h, w};  // 2 planes: Y and UV
                strides = {
                  buffer_info.color_planes[0].stride,  // Y plane stride
                  buffer_info.color_planes[1].stride,  // UV plane stride (same as Y)
                  1  // Stride within a row
                };
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM: {
                GXF_LOG_WARNING("Empty dimensions and strides for VideoFormat: %s", color_format.c_str());
              } break;
              default: {
                GXF_LOG_ERROR("Unsupported VideoFormat (%ld)", static_cast<int64_t>(buffer_info.color_format));
              } break;
            }

            return pybind11::make_tuple(
                              pybind11::cast(static_cast<void*>(v.pointer())),
                              v.size(),
                              pybind11::dtype(PythonBufferFormatString(descriptor)),
                              dimensions,
                              strides);
          })
      .def(
          "get_color_format",
          [](nvidia::gxf::VideoBuffer& v) {
            std::string color_format;
            auto buffer_info = v.video_frame_info();

            switch (buffer_info.color_format) {
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
                color_format = nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>::name;
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA: {
                color_format = nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>::name;
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12: {
                color_format = nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12>::name;
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM: {
                color_format = nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM>::name;
                GXF_LOG_WARNING("VideoFormat: %s", color_format.c_str());
              } break;
              default: {
                GXF_LOG_ERROR("Unsupported VideoFormat (%ld)", static_cast<int64_t>(buffer_info.color_format));
              } break;
            }

            return color_format;
          })
      .def(
          "get_offsets",
          [](nvidia::gxf::VideoBuffer& v) {
            std::vector<uint32_t> offsets;
            auto buffer_info = v.video_frame_info();

            // Get offsets info between planes for multi-plane buffers
            switch (buffer_info.color_format) {
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
                offsets = {0};
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12: {
                offsets = {
                  buffer_info.color_planes[0].offset,  // Y plane offset
                  buffer_info.color_planes[1].offset   // UV plane offset
                };
              } break;
              case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM: {
                GXF_LOG_WARNING("Empty offsets for VideoFormat::GXF_VIDEO_FORMAT_CUSTOM");
              } break;
              default: {
                GXF_LOG_ERROR("Unsupported VideoFormat (%ld)", static_cast<int64_t>(buffer_info.color_format));
              } break;
            }

            return offsets;
          })
      .def("storage_type", [](nvidia::gxf::VideoBuffer& v){
        return v.storage_type();
      })
      .def(
          "as_tensor",
          [](nvidia::gxf::VideoBuffer& v) {
            nvidia::gxf::Tensor* t = new nvidia::gxf::Tensor();
            auto result2 = v.moveToTensor(t);
            if (!result2) { throw pybind11::value_error(GxfResultStr(result2.error())); }
            return t;
          },
          pybind11::return_value_policy::reference);
}
