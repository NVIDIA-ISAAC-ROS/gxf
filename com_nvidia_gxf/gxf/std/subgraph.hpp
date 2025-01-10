/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_CORE_SUBGRAPH_HPP
#define NVIDIA_GXF_CORE_SUBGRAPH_HPP

#include <map>
#include <string>

#include "gxf/core/gxf.h"

#include "common/assert.hpp"
#include "common/type_name.hpp"
#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {


class Subgraph : public Component {
 public:
  gxf_result_t registerInterface(Registrar* registrar) {
    Expected<void> result;
    result &= registrar->parameter(location_, "location", "Yaml source of the subgraph", "");
    result &= registrar->parameter(
      prerequisites_, "prerequisites",
      "list of prerequisite components of the subgraph",
      "a prerequisite is a component required by the subgraph and must be satisfied"
      " before the graph is loaded",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return ToResultCode(result);
  }

 private:
  Parameter<FilePath> location_;
  Parameter<std::map<std::string, Handle<Component>>> prerequisites_;
};

template <>
struct ParameterParser<std::map<std::string, Handle<Component>>> {
  static Expected<std::map<std::string, Handle<Component>>> Parse(
    gxf_context_t context, gxf_uid_t component_uid, const char* key,
    const YAML::Node& node, const std::string& prefix) {
      if (!node.IsMap()) {
        return Unexpected{GXF_PARAMETER_PARSER_ERROR};
      }
      std::map<std::string, Handle<Component>> prerequisites;
      for (const auto& p : node) {
        const std::string k = p.first.as<std::string>();
        const auto maybe = ParameterParser<Handle<Component>>::Parse(
          context, component_uid, k.c_str(), node[k], prefix);
        if (!maybe) {
          return ForwardError(maybe);
        }
        prerequisites[k] = maybe.value();
      }
      return prerequisites;
    }
};

template <>
struct ParameterWrapper<std::map<std::string, Handle<Component>>> {
  // Wrap the value to a YAML::Node instance
  static Expected<YAML::Node> Wrap(
    gxf_context_t context,
    const std::map<std::string, Handle<Component>>& value) {
    YAML::Node node(YAML::NodeType::Map);
    for (auto &i : value) {
      auto maybe = ParameterWrapper<Handle<Component>>::Wrap(context, i.second);
      if (!maybe) {
        return ForwardError(maybe);
      }
      node[i.first] = maybe.value();
    }
    return node;
  }
};
}  // namespace  gxf
}  // namespace nvidia

#endif
