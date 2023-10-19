/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_GXF_STD_IPC_SERVER_HPP_
#define NVIDIA_GXF_GXF_STD_IPC_SERVER_HPP_

#include <functional>
#include <string>

#include "gxf/core/component.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

// interface of a request/response based server which accepts requests composed
// of a list of strings as parameters and returns a text(json) response
class IPCServer : public Component {
 public:
  typedef enum {
    kQuery = 0,
    kAction
  } ServiceType;

  // service handler for query requests
  // params:
  //   resource: on which the query is to be performed
  // return: text response that carries the result of a query
  typedef std::function<Expected<std::string>(const std::string& resource)> QueryHandler;

  // service handler for action requests
  // params:
  //  resource: on which the action is to be performed
  //  data: action data
  // return: indicator of a success or failure
  typedef std::function<Expected<void>(const std::string& resource,
                                       const std::string& data)> ActionHandler;

  // callback to handle the request received by the server
  typedef struct {
    std::string name;
    ServiceType type;
    // TODO(chunlinl): use std::variant to wrap the handler once c++17 support is added
    struct {
      QueryHandler query;
      ActionHandler action;
    } handler;
  } Service;

  // register a named service on the IPC server to handler requests from remote clients
  virtual Expected<void> registerService(const Service& service) = 0;
};

}  // namespace gxf
}  // namespace nvidia

#endif
