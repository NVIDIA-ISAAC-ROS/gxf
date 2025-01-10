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

#include "gxf/core/component.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/double_buffer_receiver.hpp"

PYBIND11_MODULE(receiver_pybind, m) {
  pybind11::class_<nvidia::gxf::Receiver>(m, "Receiver")
      .def("receive",
           [](nvidia::gxf::Receiver& r) {
             auto message = r.receive();
             if (!message || message.value().is_null()) {
               if (!message) { GXF_LOG_ERROR("No Message"); }
               if (message.value().is_null()) { GXF_LOG_ERROR("Message Null"); }
               return nvidia::gxf::Entity();
             }
             return message.value();
           })
      .def("sync", [](nvidia::gxf::Receiver &r){
        auto result = r.sync();
        if(!result){
          GXF_LOG_ERROR("Sync Failed");
          std::runtime_error(GxfResultStr(result.error()));
        }
        return;
      })
      .def("back_size", &nvidia::gxf::Receiver::back_size)
      .def("size", &nvidia::gxf::Receiver::size)
      .def("capacity", &nvidia::gxf::Receiver::capacity)
      .def("get", [](gxf_context_t context, gxf_uid_t cid, const char* name) {
          std::vector<nvidia::gxf::Receiver*> result;
          auto maybe_receivers = nvidia::gxf::CreateHandleFromString<nvidia::gxf::Receiver>(context, cid, name);
          if (!maybe_receivers) {
            // GXF_LOG_ERROR("[E%05zu] Couldn't get receivers", this->eid());
            throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
          }
          return maybe_receivers.value().get();
      },  pybind11::return_value_policy::reference);
  pybind11::class_<nvidia::gxf::DoubleBufferReceiver, nvidia::gxf::Receiver>(
      m, "DoubleBufferReceiver").def(pybind11::init<>());
}
