/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

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
        auto maybe_cuda_stream_pools = nvidia::gxf::CreateHandleFromString<nvidia::gxf::CudaStreamPool>(context, cid, name);
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
