/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/tensor.hpp"

namespace {

// Gets Python buffer format string from element type of the tensor
const char* PythonBufferFormatString(nvidia::gxf::PrimitiveType element_type) {
  switch (element_type) {
    case nvidia::gxf::PrimitiveType::kCustom:
      return "";
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
    case nvidia::gxf::PrimitiveType::kFloat32:
      return "f";
    case nvidia::gxf::PrimitiveType::kFloat64:
      return "d";
    case nvidia::gxf::PrimitiveType::kComplex64:
      return "Zf";
    case nvidia::gxf::PrimitiveType::kComplex128:
      return "Zd";
  }
  return "";
}

}  // namespace

PYBIND11_MODULE(tensor_pybind, m) {
  m.doc() = R"pbdoc(
        Python bridge for accessing tensor buffer
        -----------------------

        .. currentmodule:: pygxf

    )pbdoc";

  // Define a pybind buffer parser that converts tensors to numpy array automatically on return
  pybind11::class_<nvidia::gxf::Handle<nvidia::gxf::Tensor>>(m, "TensorHandle",
                                                             pybind11::buffer_protocol())
      .def_buffer([](nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor) -> pybind11::buffer_info {
        std::vector<int> dimensions;
        std::vector<int> strides;
        const char* format = PythonBufferFormatString(tensor->element_type());
        nvidia::gxf::Shape tensor_shape = tensor->shape();
        for (size_t i = 0; i < tensor_shape.rank(); i++) {
          dimensions.push_back(tensor_shape.dimension(i));
          strides.push_back(tensor->stride(i));
        }
        return pybind11::buffer_info(tensor->pointer(),           /* Pointer to buffer */
                                     tensor->bytes_per_element(), /* Size of one scalar */
                                     format, /* Python struct-style format descriptor */
                                     tensor_shape.rank(), /* Number of dimensions */
                                     dimensions,          /* Buffer dimensions */
                                     strides              /* Stride dimensions */
        );
      });

  m.def("as_tensor", [](uint64_t context_idx, gxf_uid_t eid, const char* tensor_field) {
    const gxf_context_t context = reinterpret_cast<gxf_context_t>(context_idx);
    auto entity = nvidia::gxf::Entity::Shared(context, eid);
    auto maybe_tensor = entity.value().get<nvidia::gxf::Tensor>(tensor_field);
    if (!maybe_tensor) { throw pybind11::value_error("Field with matching name does not exist"); }
    return pybind11::array(pybind11::cast(maybe_tensor.value()));
  });

  pybind11::class_<nvidia::gxf::Tensor>(m, "Tensor", pybind11::buffer_protocol())
      .def_buffer([](nvidia::gxf::Tensor& tensor) -> pybind11::buffer_info {
        std::vector<int> dimensions;
        std::vector<int> strides;
        const char* format = PythonBufferFormatString(tensor.element_type());
        nvidia::gxf::Shape tensor_shape = tensor.shape();
        for (size_t i = 0; i < tensor_shape.rank(); i++) {
          dimensions.push_back(tensor_shape.dimension(i));
          strides.push_back(tensor.stride(i));
        }
        return pybind11::buffer_info(tensor.pointer(),           /* Pointer to buffer */
                                     tensor.bytes_per_element(), /* Size of one scalar */
                                     format, /* Python struct-style format descriptor */
                                     tensor_shape.rank(), /* Number of dimensions */
                                     dimensions,          /* Buffer dimensions */
                                     strides              /* Stride dimensions */
        );
      })
      .def("get_tensor_info",
           [](nvidia::gxf::Tensor& t) {
             auto rank = t.rank();
             std::vector<int32_t> dims;
             std::vector<int32_t> strides;
             void* buffer_ptr = static_cast<void*>(t.pointer());
             std::string descriptor = PythonBufferFormatString(t.element_type());
             for (uint i = 0; i < rank; i++) {
               strides.push_back(t.stride(i));
               dims.push_back(t.shape().dimension(i));
             }
            //  Pybind does not recognize Zf / Zd format strings
             if (descriptor == "Zf") descriptor = "complex64";
             if (descriptor == "Zd") descriptor = "complex128";
             return pybind11::make_tuple(pybind11::cast(buffer_ptr), t.size(),
                                         pybind11::dtype(descriptor), rank, dims, strides);
           })
      .def("shape", &nvidia::gxf::Tensor::shape)
      .def("element_type", &nvidia::gxf::Tensor::element_type)
      .def("storage_type", &nvidia::gxf::Tensor::storage_type)
      .def("reshape",
           [](nvidia::gxf::Tensor& t, nvidia::gxf::TensorDescription& td,
              nvidia::gxf::Allocator* allocator) {
             auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                 allocator->context(), allocator->cid());
             if (!allocator_handle) {
               throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
             }
             nvidia::gxf::Expected<void> result;
             if (sizeof(td.strides)==0) {
               result = t.reshapeCustom(td.shape, td.element_type, td.bytes_per_element,
                                        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                        td.storage_type, allocator_handle.value());
             } else {
               result = t.reshapeCustom(td.shape, td.element_type, td.bytes_per_element, td.strides,
                                        td.storage_type, allocator_handle.value());
             }
             if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
             return;
           })
      .def("reshape_custom",
           [](nvidia::gxf::Tensor& t, const nvidia::gxf::Shape& shape,
              nvidia::gxf::PrimitiveType element_type, uint64_t bytes_per_element,
              std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides,
              nvidia::gxf::MemoryStorageType storage_type, nvidia::gxf::Allocator* allocator) {
             auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                 allocator->context(), allocator->cid());
             if (!allocator_handle) {
               throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
             }
             nvidia::gxf::Expected<void> result;
             if (strides.size() == 0) {
               result = t.reshapeCustom(shape, element_type, bytes_per_element,
                                        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                        storage_type, allocator_handle.value());
             } else {
               result = t.reshapeCustom(shape, element_type, bytes_per_element, strides,
                                        storage_type, allocator_handle.value());
             }
             if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
             return;
           })
      .def(
          "get_tensor_description",
          [](nvidia::gxf::Tensor& t) {
            nvidia::gxf::TensorDescription td;
            td.shape = t.shape();
            td.element_type = t.element_type();
            td.bytes_per_element = t.bytes_per_element();
            td.storage_type = t.storage_type();
            return td;
          },
          pybind11::return_value_policy::reference)
      .def(
          "add_np_array_as_tensor_to_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr, pybind11::array array,
             nvidia::gxf::Allocator * allocator) {
            // add a gxf::Tensor to the entity
            auto result = e.add<nvidia::gxf::Tensor>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            auto t = result.value().get();

            // calculate relevant info
            std::array<int32_t, nvidia::gxf::Shape::kMaxRank> shape;
            auto array_shape = array.shape();
            for (auto i = 0; i < array.ndim(); i++) { shape[i] = *(array_shape + i); }
            std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides;
            auto array_strides = array.strides();
            for (auto i = 0; i < array.ndim(); i++) { strides[i] = *(array_strides + i); }
            nvidia::gxf::PrimitiveType element_type;
            if (pybind11::str(array.dtype()).equal(pybind11::str("int8")))
              element_type = nvidia::gxf::PrimitiveType::kInt8;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint8")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("int16")))
              element_type = nvidia::gxf::PrimitiveType::kInt16;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint16")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("int32")))
              element_type = nvidia::gxf::PrimitiveType::kInt32;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint32")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("int64")))
              element_type = nvidia::gxf::PrimitiveType::kInt64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint64")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("float32")))
              element_type = nvidia::gxf::PrimitiveType::kFloat32;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("float64")))
              element_type = nvidia::gxf::PrimitiveType::kFloat64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("complex64")))
              element_type = nvidia::gxf::PrimitiveType::kComplex64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("complex128")))
              element_type = nvidia::gxf::PrimitiveType::kComplex128;
            else
              element_type = nvidia::gxf::PrimitiveType::kCustom;

            // create a tensor description
            nvidia::gxf::TensorDescription td{
                name,
                nvidia::gxf::MemoryStorageType::kHost,
                nvidia::gxf::Shape(shape, static_cast<uint32_t>(array.ndim())),
                element_type,
                static_cast<uint64_t>(array.itemsize()),
                strides};

            // reshape the tensor
            auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                allocator->context(), allocator->cid());
            if (!allocator_handle) {
              throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
            }

            nvidia::gxf::Expected<void> reshape_result;
            if (sizeof(td.strides) == 0) {
              reshape_result = t->reshapeCustom(td.shape, td.element_type, td.bytes_per_element,
                                                nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                                td.storage_type, allocator_handle.value());
            } else {
              reshape_result = t->reshapeCustom(td.shape, td.element_type, td.bytes_per_element,
                                                td.strides, td.storage_type, allocator_handle.value());
            }
            if (!reshape_result) {
              throw pybind11::value_error(GxfResultStr(reshape_result.error()));
            }

            // copy the data from numpy array to this tensor
            std::memcpy(t->pointer(), array.data(), array.nbytes());
            return t;
          },
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto result = e.add<nvidia::gxf::Tensor>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            return result.value().get();
          },
          pybind11::return_value_policy::reference)
      .def_static(
          "get_from_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto maybe_tensor = e.get<nvidia::gxf::Tensor>(name);
            if (!maybe_tensor) {
              GXF_LOG_ERROR("Error getting tensor called %s", name);
              throw pybind11::value_error("error getting tensor");
            }
            return maybe_tensor.value().get();
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def(
          "find_all_from_entity",
          [](nvidia::gxf::Entity& e) {
            nvidia::FixedVector<nvidia::gxf::Handle<nvidia::gxf::Tensor>, kMaxComponents>
                components;
            auto maybe_tensors = e.findAll<nvidia::gxf::Tensor>(components);
            if (!maybe_tensors) { throw pybind11::value_error("error getting tensors"); }
            std::vector<nvidia::gxf::Tensor*> result;
            for (uint i = 0; i < components.size(); i++) {
              auto maybe_tensor = components.at(i);
              if (!maybe_tensor) { throw std::runtime_error("Error getting tensor"); }
              result.push_back(maybe_tensor.value().get());
            }
            return result;
          },
          pybind11::return_value_policy::reference);

  pybind11::enum_<nvidia::gxf::PrimitiveType>(m, "PrimitiveType")
      .value("kCustom", nvidia::gxf::PrimitiveType::kCustom)
      .value("kInt8", nvidia::gxf::PrimitiveType::kInt8)
      .value("kUnsigned8", nvidia::gxf::PrimitiveType::kUnsigned8)
      .value("kInt16", nvidia::gxf::PrimitiveType::kInt16)
      .value("kUnsigned16", nvidia::gxf::PrimitiveType::kUnsigned16)
      .value("kInt32", nvidia::gxf::PrimitiveType::kInt32)
      .value("kUnsigned32", nvidia::gxf::PrimitiveType::kUnsigned32)
      .value("kInt64", nvidia::gxf::PrimitiveType::kInt64)
      .value("kUnsigned64", nvidia::gxf::PrimitiveType::kUnsigned64)
      .value("kFloat32", nvidia::gxf::PrimitiveType::kFloat32)
      .value("kFloat64", nvidia::gxf::PrimitiveType::kFloat64)
      .value("kComplex64", nvidia::gxf::PrimitiveType::kComplex64)
      .value("kComplex128", nvidia::gxf::PrimitiveType::kComplex128);

  pybind11::enum_<nvidia::gxf::MemoryStorageType>(m, "MemoryStorageType")
      .value("kHost", nvidia::gxf::MemoryStorageType::kHost)
      .value("kDevice", nvidia::gxf::MemoryStorageType::kDevice)
      .value("kSystem", nvidia::gxf::MemoryStorageType::kSystem);

  pybind11::class_<nvidia::gxf::TensorDescription>(m, "TensorDescription")
      .def(pybind11::init([](std::string name, nvidia::gxf::MemoryStorageType storage_type,
                             nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
                             uint64_t bytes_per_element, std::vector<uint64_t> strides) {
             if (strides.size() == 0) {
               return std::unique_ptr<nvidia::gxf::TensorDescription>(
                   new nvidia::gxf::TensorDescription{name, storage_type, shape, element_type,
                                                      bytes_per_element});
             } else {
               std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides_array;
               std::copy_n(strides.begin(), strides.size(), strides_array.begin());
               return std::unique_ptr<nvidia::gxf::TensorDescription>(
                   new nvidia::gxf::TensorDescription{name, storage_type, shape, element_type,
                                                      bytes_per_element, strides_array});
             }
           }),
           "Description of nvidia::gxf::Tensor", pybind11::arg("name"),
           pybind11::arg("storage_type"), pybind11::arg("shape"), pybind11::arg("element_type"),
           pybind11::arg("bytes_per_element"), pybind11::arg("strides") = std::vector<uint64_t>{},
           pybind11::return_value_policy::reference)
      .def_readwrite("name", &nvidia::gxf::TensorDescription::name)
      .def_readwrite("storage_type", &nvidia::gxf::TensorDescription::storage_type)
      .def_readwrite("shape", &nvidia::gxf::TensorDescription::shape)
      .def_readwrite("element_type", &nvidia::gxf::TensorDescription::element_type)
      .def_readwrite("bytes_per_element", &nvidia::gxf::TensorDescription::bytes_per_element)
      .def_property(
          "strides",
          [](nvidia::gxf::TensorDescription& t) {
            if (t.strides) {
              return t.strides.value();
            } else {
              return std::array<uint64_t, nvidia::gxf::Shape::kMaxRank>{};
            }
          },
          [](nvidia::gxf::TensorDescription& t, std::vector<uint64_t> strides_) {
            throw pybind11::value_error("Setting stride not supported yet from python");
          });

  pybind11::class_<nvidia::gxf::Shape>(m, "Shape")
      .def(pybind11::init())
      .def(pybind11::init([](std::vector<int32_t> dims) {
             std::array<int32_t, nvidia::gxf::Shape::kMaxRank> dims_array;
             std::copy_n(dims.begin(), dims.size(), dims_array.begin());
             return std::unique_ptr<nvidia::gxf::Shape>(
                 new nvidia::gxf::Shape(dims_array, dims.size()));
           }),
           pybind11::return_value_policy::reference)
      .def("rank", &nvidia::gxf::Shape::rank)
      .def("size", &nvidia::gxf::Shape::size)
      .def("dimension", &nvidia::gxf::Shape::dimension)
      .def(pybind11::self == pybind11::self);
}
