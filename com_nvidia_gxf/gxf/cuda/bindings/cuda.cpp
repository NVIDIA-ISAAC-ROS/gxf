/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"


template <typename S>
nvidia::gxf::Expected<nvidia::gxf::Handle<S>> getHandle(gxf_context_t context, gxf_uid_t cid, const char* name) {
  gxf_uid_t eid;
  std::string component_name;
  const std::string tag = std::string(name);
  const size_t pos = tag.find('/');

  if (pos == std::string::npos) {
    // Get the entity of this component
    const gxf_result_t result_1 = GxfComponentEntity(context, cid, &eid);
    if (result_1 != GXF_SUCCESS) {
      GXF_LOG_ERROR("%s", GxfResultStr(result_1));
      throw std::runtime_error(GxfResultStr(result_1));
    }
    component_name = tag;
  } else {
    // Split the tag into entity and component name
    const std::string entity_name = tag.substr(0, pos);
    component_name = tag.substr(pos + 1);
    // Search for the entity
    const gxf_result_t result_1 = GxfEntityFind(context, entity_name.c_str(), &eid);
    if (result_1 != GXF_SUCCESS) {
      GXF_LOG_ERROR(
          "[E%05zu] Could not find entity '%s' while parsing parameter '%s' of component %zu",
          eid, entity_name.c_str(), tag.c_str(), cid);
      throw std::runtime_error(GxfResultStr(result_1));
    }
  }
  // Get the type id of the component we are are looking for.
  gxf_tid_t tid;
  const gxf_result_t result_1 = GxfComponentTypeId(context, nvidia::TypenameAsString<S>(), &tid);
  if (result_1 != GXF_SUCCESS) {
    GXF_LOG_ERROR("%s", GxfResultStr(result_1));
    throw std::runtime_error(GxfResultStr(result_1));
  }
  gxf_uid_t cid2;
  // Find the component in the indicated entity
  const gxf_result_t result_2 =
      GxfComponentFind(context, eid, tid, component_name.c_str(), nullptr, &cid2);
  if (result_2 != GXF_SUCCESS) {
    GXF_LOG_ERROR("%s", GxfResultStr(result_2));
    throw std::runtime_error(GxfResultStr(result_2));
  }


  auto handle = nvidia::gxf::Handle<S>::Create(context, cid2);
  return handle;
}

PYBIND11_MODULE(cuda_pybind, m) {
  pybind11::class_<nvidia::gxf::CudaStreamPool>(m, "CudaStreamPool")
      .def("release_stream",
           [](nvidia::gxf::CudaStreamPool& cuda_stream_pool, gxf_uid_t cid) {
             auto maybe_handle = nvidia::gxf::Handle<nvidia::gxf::CudaStream>::Create(
                 cuda_stream_pool.context(), cid);
             if (!maybe_handle) { throw pybind11::value_error(GxfResultStr(maybe_handle.error())); }
             cuda_stream_pool.releaseStream(maybe_handle.value());
             return;
           })
      .def("allocate_stream", [](nvidia::gxf::CudaStreamPool& cuda_stream_pool) {
        auto maybe_stream = cuda_stream_pool.allocateStream();
        if (!maybe_stream) { throw pybind11::value_error(GxfResultStr(maybe_stream.error())); }
        auto stream = maybe_stream.value();
        return pybind11::make_tuple(stream.cid(), stream.get());
      })
      .def("get", [](gxf_context_t context, gxf_uid_t cid, const char* name) {
        auto maybe_cuda_stream_pools = getHandle<nvidia::gxf::CudaStreamPool>(context, cid, name);
        if (!maybe_cuda_stream_pools) {
          GXF_LOG_ERROR("%s", GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
          throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
        }
        return maybe_cuda_stream_pools.value().get();
      }, pybind11::return_value_policy::reference);

  pybind11::class_<nvidia::gxf::CudaStream>(m, "CudaStream");

  pybind11::class_<nvidia::gxf::CudaStreamId>(m, "CudaStreamId")
      .def_readwrite("stream_cid", &nvidia::gxf::CudaStreamId::stream_cid)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto result = e.add<nvidia::gxf::CudaStreamId>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            return result.value().get();
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def_static(
          "get_from_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto maybe_cuda_stream_id = e.get<nvidia::gxf::CudaStreamId>(name);
            if (!maybe_cuda_stream_id) {
              throw pybind11::value_error(GxfResultStr(maybe_cuda_stream_id.error()));
            }
            return maybe_cuda_stream_id.value().get();
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference);
}
