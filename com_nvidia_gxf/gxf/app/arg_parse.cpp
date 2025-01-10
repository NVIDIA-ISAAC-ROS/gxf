/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <complex>
#include <string>
#include <vector>

#include "gxf/app/arg_parse.hpp"

namespace nvidia {
namespace gxf {

std::vector<Arg> filterProxyComponents(std::vector<Arg>& args) {
  std::vector<Arg> arg_list;
  for (auto it = args.begin(); it != args.end();) {
    if (it->parameter_type() == GXF_PARAMETER_TYPE_HANDLE && it->handle_uid() == kUnspecifiedUid) {
      arg_list.push_back(*it);
      it = args.erase(it);
    } else {
      ++it;
    }
  }
  return arg_list;
}

Expected<void> applyArg(Handle<Component> component, const Arg& arg) {
  Expected<void> result = Unexpected{GXF_ARGUMENT_INVALID};
  if (!arg.has_value()) {
    GXF_LOG_ERROR("Arg [%s] does not have any value", arg.key());
    return result;
  }

  auto maybe_info = component->getParameterInfo(arg.key());
  if (!maybe_info) {
    GXF_LOG_ERROR("Failed to get parameter info with key [%s] from component [%s] with error [%s]",
                  arg.key(), component->name(), GxfResultStr(maybe_info.error()));
    return result;
  }

  auto param_info = maybe_info.value();

  // TODO(chandrahasj) Fix this check
  // if (param_info.type == GXF_PARAMETER_TYPE_HANDLE && param_info.handle_tid != arg.handle_tid())
  // {
  //   GXF_LOG_ERROR("Arg [%s] handle tid 0x%016zx%016zx does not match info from component type
  //   [%s] which is"
  //                 " 0x%016zx%016zx ", arg.key(), arg.handle_tid().hash1, arg.handle_tid().hash2,
  //                 component->type_name(), param_info.handle_tid.hash1,
  //                 param_info.handle_tid.hash2);
  //   return result;
  // }

  if (param_info.rank != arg.rank()) {
    GXF_LOG_ERROR(
        "Arg [%s] rank [%d] does not match the info from component type [%s] which is [%d]",
        arg.key(), arg.rank(), component->type_name(), param_info.rank);
    return result;
  }

  for (auto i = 0; i < param_info.rank; ++i) {
    if (param_info.shape[i] != arg.shape().at(i)) {
      GXF_LOG_ERROR("Arg [%s] shape does not match the info from component type [%s]", arg.key(),
                    component->type_name());
      return result;
    }
  }

  result = component->parseParameter(arg.key(), arg.yaml_node());
  if (!result) {
    GXF_LOG_ERROR("Failed to set arg [%s] of type [%s]", arg.key(), arg.arg_type_name().c_str());
  }

  return result;
}

Expected<Arg> findArg(const std::vector<Arg>& args, const std::string& key,
  const gxf_parameter_type_t type) {
  auto it = std::find_if(args.begin(), args.end(),
    [&](const Arg& arg) { return strcmp(arg.key(), key.c_str()) == 0; });
  if (it == args.end()) {
    GXF_LOG_DEBUG("Cannot find arg with key: %s from provided arg list", key.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  if (it->parameter_type() !=  type) {
    GXF_LOG_ERROR("Arg with key: %s is of type: %d, instead of %d",
      key.c_str(), it->parameter_type(), type);
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return *it;
}

}  // namespace gxf
}  // namespace nvidia
