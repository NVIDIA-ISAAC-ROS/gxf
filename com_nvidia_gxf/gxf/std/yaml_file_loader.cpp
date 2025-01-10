/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/yaml_file_loader.hpp"

#include <inttypes.h>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/assert.hpp"
#include "gxf/core/common_expected_macro.hpp"
#include "gxf/core/parameter_storage.hpp"
#include "gxf/core/type_registry.hpp"

namespace nvidia {
namespace gxf {

namespace {

constexpr char kAttributeName[] = "name";
constexpr char kAttributeType[] = "type";
constexpr char kAttributeTarget[] = "target";
constexpr char kAttributeComponents[] = "components";
constexpr char kAttributeParameters[] = "parameters";
constexpr char kAttributeInterfaces[] = "interfaces";
constexpr char kAttributePrerequisites[] = "prerequisites";
constexpr char kAttributeStartOrderName[] = "start_order";
constexpr char kAttributeEntityGroups[] = "EntityGroups";
constexpr char kComponentTypeSubgraph[] = "nvidia::gxf::Subgraph";

constexpr size_t kNumParamString = 4;
struct ParameterOverrides {
  std::string entity;
  std::string component;
  std::string parameter;
  std::string value;
};

static const YAML::Node NullYamlNode = YAML::Node(YAML::NodeType::Null);

// Converts std::vector to FixedVector
template <typename T>
Expected<void> StdVectorToFixedVector(std::vector<T>& src, FixedVectorBase<T>& dst) {
  for (size_t i = 0; i < src.size(); i++) {
    auto result = dst.push_back(std::move(src[i]));
    if (!result) { return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
  }
  return Success;
}

Expected<bool> isSubgraph(gxf_context_t context, gxf_uid_t cid) {
  gxf_tid_t tid = GxfTidNull();
  auto code = GxfComponentType(context, cid, &tid);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component type");
    return Unexpected{GXF_FAILURE};
  }
  const char* component_type_name = nullptr;
  code = GxfComponentTypeName(context, tid, &component_type_name);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component type name");
    return Unexpected{GXF_FAILURE};
  }
  return (std::string(component_type_name) == kComponentTypeSubgraph);
}

// String split
Expected<void> split(std::string input_string, std::string delimiter,
    std::vector<std::string>& result) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;

    while ((pos_end = input_string.find(delimiter, pos_start)) != std::string::npos) {
        token = input_string.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        result.push_back(token);
    }

    result.push_back(input_string.substr(pos_start));
    return Success;
}

Expected<void> ParseParameterOverride(const char* inputStr[],
    const uint32_t num_overrides,
    std::vector<ParameterOverrides>& parameter_override) {
    for (uint32_t idx = 0; idx < num_overrides; idx++) {
      // Result in order of value first and then entity/component/parameter
      std::vector<std::string> value;
      struct ParameterOverrides param_out;
      // First parse for the combination of "component+parameter" and "value" which is
      // delimited by '='
      split(inputStr[idx], "=", value);
      // Last split chunk will have a value delimited by '='
      // Now get the entity, component and parameter by using the first split chunk
      std::vector<std::string> params;
      split(value[0], "/", params);
      if (params.size() == (kNumParamString - 1)) {
        param_out.entity = params[0];
        param_out.component = params[1];
        param_out.parameter = params[2];
        param_out.value = value[1];
        parameter_override.push_back(param_out);
      } else {
        GXF_LOG_ERROR("Override parameter string is incorrect:%s", inputStr[idx]);
        return Unexpected{GXF_ARGUMENT_INVALID};
      }
    }
    return Success;
}

// update the parameter value from the YAML node if any parameter override
// found, otherwise return the original YAML node for component parameters
Expected<YAML::Node> performParameterOverride(
  YAML::Node component,
  std::vector<ParameterOverrides>& parameter_override,
  const char* entity_name) {
  auto yaml_param = component[kAttributeParameters];
  std::string component_type_name = "";
  if (const auto& type_yaml = component[kAttributeType]) {
    component_type_name = component[kAttributeType].as<std::string>();
  }
  for (size_t i = 0; i < parameter_override.size(); i++) {
    const auto& param = parameter_override.at(i);
    bool found = false;
    if (param.entity.compare(entity_name) == 0) {
      if (const auto& name_yaml = component[kAttributeName]) {
        if (param.component.compare(name_yaml.as<std::string>()) == 0) {
          found = true;
        }
      }

      if (!found && param.component.compare(component_type_name) == 0) {
        found = true;
      }

      if (found == true) {
        if (!yaml_param.IsMap()) {
          GXF_LOG_ERROR("Override: Could not parse parameters for - not a map");
          return Unexpected{GXF_INVALID_DATA_FORMAT};
        }
        GXF_LOG_INFO("Parameter(%s) is override with value (%s)",
                      param.parameter.c_str(), param.value.c_str());
        yaml_param[param.parameter] = YAML::Load(param.value);
      }
    }
  }
  return yaml_param;
}

}  // namespace

Expected<void> YamlFileLoader::loadFromFile(
    gxf_context_t context, const std::string& filename,
    const std::string& entity_prefix,
    const char* parameters_override_string[],
    const uint32_t num_overrides,
    gxf_uid_t parent_eid, const YAML::Node& prerequisites) {
  std::string path = root_.empty() || filename.at(0) == '/' ?
      filename : root_ + "/" + filename;
  try {
    GXF_LOG_INFO("Loading GXF entities from YAML file '%s'...", path.c_str());
    FixedVector<YAML::Node, kMaxEntities> nodes;
    auto yaml_nodes = YAML::LoadAllFromFile(path);
    return StdVectorToFixedVector(yaml_nodes, nodes)
        .and_then([&](){
          return load(context, nodes, entity_prefix, parent_eid,
                      parameters_override_string, num_overrides, prerequisites);
        });
  } catch (YAML::Exception& e) {
    GXF_LOG_ERROR("Could not load yaml file '%s':\n%s", path.c_str(), e.what());
    return Unexpected{GXF_FAILURE};
  }
}

Expected<void> YamlFileLoader::loadFromString(
    gxf_context_t context, const std::string& text,
    const std::string& entity_prefix,
    const char* parameters_override_string[],
    const uint32_t num_overrides) {
  try {
    GXF_LOG_INFO("Loading GXF entities from string...");
    FixedVector<YAML::Node, kMaxEntities> nodes;
    auto yaml_nodes = YAML::LoadAll(text);
    return StdVectorToFixedVector(yaml_nodes, nodes)
        .and_then([&](){
          return load(context, nodes, entity_prefix, kNullUid,
                      parameters_override_string, num_overrides, NullYamlNode);
        });
  } catch (YAML::Exception& e) {
    GXF_LOG_ERROR("Could not load yaml from string:\n%s", e.what());
    return Unexpected{GXF_FAILURE};
  }
}

Expected<void> YamlFileLoader::load(
    gxf_context_t context,
    const FixedVectorBase<YAML::Node>& nodes,
    std::string entity_prefix,  //< prefix to be added to the entities in the current graph
    gxf_uid_t parent_eid,  //< uid of the entity which loads the current graph as a subgraph
    const char* parameters_override_string[],
    const uint32_t num_overrides,
    const YAML::Node& prerequisites  //< fulfillment of the prerequisites of the current graph
  ) {
  if (!entity_prefix.empty()) {
    entity_prefix = entity_prefix + ".";
  }

  // Parameter override
  std::vector<ParameterOverrides> parameter_override;
  if (num_overrides != 0 && parameters_override_string != nullptr) {
    ParseParameterOverride(parameters_override_string, num_overrides, parameter_override);
  } else if (num_overrides != 0 || parameters_override_string != nullptr) {
    GXF_LOG_ERROR("Invalid parameter override config");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  std::vector<std::string> start_order;
  std::string current_graph_file{};
  std::vector<YAML::Node> all_nodes;
  std::map<gxf_uid_t, YAML::Node> all_components;
  size_t num_entities = 0;
  int32_t start_order_node_count = 0;

  try {
    // Find whether there is an entity specified that specifies the start order of the entities.
    for (size_t i = 0; i < nodes.size(); i++) {
      const auto& node = nodes.at(i).value();
      if (node.IsNull()) { continue; }

      if (!node.IsMap()) {
        GXF_LOG_ERROR("YAML node is not a map '%s'", node.as<std::string>().c_str());
        return Unexpected{GXF_INVALID_DATA_FORMAT};
      }
      if (const auto& start_order_node = node[kAttributeStartOrderName]) {
        start_order = start_order_node.as<std::vector<std::string>>();
        for (auto& start : start_order) {
          start = entity_prefix + start;
        }
        start_order_node_count++;
      }
      std::string entity_name;
      if (const auto& name_yaml = node[kAttributeName]) {
        entity_name = entity_prefix + name_yaml.as<std::string>();
      }
      // Do not consider start_order entity in total no of entities
      // Start order entity doesn't have components
      if (entity_name != "" && node[kAttributeComponents]) {
        num_entities++;
      }
      if (start_order_node_count > 1) {
        GXF_LOG_ERROR("Start order entities specified more than once");
        return Unexpected{GXF_FAILURE};
      }
    }

    auto start_order_size = start_order.size();
    if (start_order_size) {
      // Move all the entities into all_nodes as per start order
      for (size_t start_order_index = 0; start_order_index < start_order_size;
           start_order_index++) {
        bool is_entity_present = false;
        for (size_t j = 0; j < nodes.size(); j++) {
          const auto& node = nodes.at(j).value();
          std::string entity_name;
          if (const auto& name_yaml = node[kAttributeName]) {
            entity_name = entity_prefix + name_yaml.as<std::string>();
          }
          if (start_order.at(start_order_index) == entity_name) {
            all_nodes.push_back(std::move(node));
            is_entity_present = true;
          }
        }
        if (is_entity_present == false) {
          GXF_LOG_WARNING("Specified entity in start order is invalid - %s",
                          start_order.at(start_order_index).c_str());
        }
      }

      if (start_order_size != num_entities) {
        GXF_LOG_WARNING("Mismatch in the number of entities specified in the start order and the" \
                        " actual entities - start_order_size = %ld, num_entities = %ld\n Trying " \
                        "to use missing entities from the yaml order",
                        start_order_size, num_entities);
      }
      if (start_order_size > num_entities) {
        GXF_LOG_ERROR("Start order entities cannot be greater than the number of actual " \
                      "entities - start_order_size = %ld, num_entities = %ld", start_order_size,
                      num_entities);
        return Unexpected{GXF_FAILURE};
      }

      // Get all the unnamed entities. These are mostly the connection components.
      // Also check whether the start order list has omitted any entity. If yes,
      // then start the entity as per the yaml order.
      for (size_t i = 0; i < nodes.size(); i++) {
        const auto& node = nodes.at(i).value();
        std::string entity_name;
        if (const auto& name_yaml = node[kAttributeName]) {
          entity_name = entity_prefix + name_yaml.as<std::string>();
        }
        // Mostly connection entities
        if ((entity_name == "") && node[kAttributeComponents]) {
            all_nodes.push_back(std::move(node));
        } else {
          // Entities not listed in the start order
          bool is_entity_present = false;
          for (size_t start_order_index = 0; start_order_index < start_order_size;
               start_order_index++) {
            if (start_order.at(start_order_index) == entity_name) {
              is_entity_present = true;
            }
          }
          if ((is_entity_present == false) && node[kAttributeComponents])
            all_nodes.push_back(std::move(node));
        }
      }
    } else {
      // If no start order is specified then just move the entities to all_nodes
      for (size_t i = 0; i < nodes.size(); i++) {
        const auto& node = nodes.at(i).value();
        all_nodes.push_back(std::move(node));
      }
    }

    std::map<gxf_uid_t, YAML::Node> subgraph_entities;  //< subgraph nodes
    // iterate all the nodes and instantiate the entities with components inside.
    // Also instantiate all the components except for subgraphs.
    for (size_t i = 0; i < all_nodes.size(); i++) {
      const auto& node = all_nodes.at(i);
      // TODO(yangl) switch to enforcing unique name of components and use name for indexing
      if (node[kAttributeComponents]) {
        // Get the entity name
        std::string entity_name;
        if (const auto& name_yaml = node[kAttributeName]) {
          entity_name = entity_prefix + name_yaml.as<std::string>();
        }

        const auto maybe_eid = findOrCreateEntity(context, entity_name);
        if (!maybe_eid) { return gxf::ForwardError(maybe_eid); }
        const gxf_uid_t eid = maybe_eid.value();
        const auto& components = node[kAttributeComponents];
        if (!components.IsSequence() && !components.IsNull()) {
          GXF_LOG_ERROR("Components must be Sequence");
          return Unexpected{GXF_FAILURE};
        }

        // first check if the entity is a subgraph entity
        bool is_subgraph = false;
        for (auto component : components) {
          if (const auto& type_yaml = component[kAttributeType]) {
            const std::string component_type_name = type_yaml.as<std::string>();
            if (component_type_name == kComponentTypeSubgraph) {
              is_subgraph = true;
              break;
            }
          } else if (const auto& name_yaml = component[kAttributeName])  {
            const std::string component_name = name_yaml.as<std::string>();
            const auto maybe = findComponent(context, eid, component_name.c_str());
            if (!maybe) {
              GXF_LOG_ERROR("Could not find component with name '%s' in entity.",
                            component_name.c_str());
              return ForwardError(maybe);
            }
            auto code = isSubgraph(context, maybe.value());
            if (!code) {
              return Unexpected{GXF_FAILURE};
            }
            if (code.value()) {
              is_subgraph = true;
              break;
            }
          }
        }
        if (is_subgraph) {
          subgraph_entities[eid] = node;
          // we'll parse subgraph node later
          continue;
        }

        const int64_t num_components = components.size();
        for (int64_t i = 0; i < num_components; i++) {
          const auto& component = components[i];
          gxf_uid_t component_handle;
          // instantiate all the new components defined in the yaml
          if (const auto& type_yaml = component[kAttributeType]) {
            // add new component
            const std::string component_type_name = type_yaml.as<std::string>();
            const auto maybe = addComponent(context, eid, component_type_name.c_str());
            if (!maybe) {
              GXF_LOG_ERROR("Could not add component of type '%s' to entity.",
                            component_type_name.c_str());
              return ForwardError(maybe);
            }
            component_handle = maybe.value();
            // set name of component as a parameter
            if (const auto& name_yaml = component[kAttributeName]) {
              const gxf_result_t code = GxfParameterSetStr(
                  context, component_handle, kInternalNameParameterKey,
                  name_yaml.as<std::string>().c_str());
              if (code != GXF_SUCCESS) { return Unexpected{code}; }
            }
          } else if (const auto& name_yaml = component[kAttributeName]) {
            // find the component for parameter overriding
            const std::string component_name = name_yaml.as<std::string>();
            const auto maybe = findComponent(context, eid, component_name.c_str());
            if (!maybe) {
              GXF_LOG_ERROR("Could not find component with name '%s' in entity.",
                            component_name.c_str());
              return ForwardError(maybe);
            }
            component_handle = maybe.value();
          } else {
            GXF_LOG_ERROR("If the type of a component is not specified it is interpreted as "
                            "setting parameters. In that case the component name is mandatory.");
            return Unexpected{GXF_FAILURE};
          }
          auto params = performParameterOverride(
            component, parameter_override, entity_name.c_str());
          if (params) {
            all_components.insert({component_handle, params.value()});
          }
        }
      }
    }

    // populates the prerequisites if there is any, before loading subgraphs in
    // case that some of them are also passed to those subgraphs.
    for (size_t i = 0; i < nodes.size(); i++) {
      const auto& node = nodes.at(i).value();
      if (node[kAttributePrerequisites]) {
        std::string entity_name = entity_prefix + kAttributePrerequisites;
        // create the prerequisites entity which only contains component mapping
        const auto maybe_eid = findOrCreateEntity(context, entity_name);
        if (!maybe_eid) {
          return gxf::ForwardError(maybe_eid);
        }
        // materialize the prerequisites by adding them to the interface
        if (!prerequisites.IsMap() || prerequisites.IsNull()) {
          GXF_LOG_WARNING("Prerequisites of subgraph %s not satisfied, be cautious!",
                          entity_prefix.c_str());
        } else {
          for (auto item : prerequisites) {
            std::string name = item.first.as<std::string>();
            std::string tag = item.second.as<std::string>();
            const size_t pos = entity_prefix.rfind('.', entity_prefix.length()-2);
            std::string parent_entity_prefix =
            (pos == std::string::npos ? std::string():entity_prefix.substr(0, pos+1));
            auto result = addComponentToInterface(
              context, maybe_eid.value(), parent_entity_prefix, name, tag);
            if (!result) {
              return gxf::ForwardError(result);
            }
          }
        }
      }
    }

    // instantiate the subgraph components in subgraph entities and perform the
    // yaml file loading
    for (auto &subgraph_iterator : subgraph_entities) {
      gxf_uid_t eid = subgraph_iterator.first;
      gxf_uid_t cid = kNullUid;
      const char* subgraph_entity_name = nullptr;
      if (GXF_SUCCESS != GxfEntityGetName(context, eid, &subgraph_entity_name)) {
          subgraph_entity_name = std::to_string(eid).c_str();
      }
      // go through the components, if there is any subgraph, let's perform the
      // loading here so that all the prerequisites can be correctly applied.
      const auto components = subgraph_iterator.second[kAttributeComponents];
      for (auto &component : components) {
        const auto& type_yaml = component[kAttributeType];
        const auto& name_yaml = component[kAttributeName];
        const auto& param_yaml = component[kAttributeParameters];

        if (type_yaml && type_yaml.as<std::string>() == kComponentTypeSubgraph) {
          // create a new component instance for the subgraph
          const auto maybe = addComponent(context, eid, kComponentTypeSubgraph);
          if (!maybe) {
            GXF_LOG_ERROR("Could not add subgraph component");
            return ForwardError(maybe);
          }
          cid = maybe.value();
          // set name of the subgraph component as a parameter
          if (name_yaml) {
            const gxf_result_t code = GxfParameterSetStr(
                context, cid, kInternalNameParameterKey,
                name_yaml.as<std::string>().c_str());
            if (code != GXF_SUCCESS) { return Unexpected{code}; }
          }
        } else if (!type_yaml && name_yaml) {
          // check if the subgraph is being loaded through a parameter file
          const std::string component_name = name_yaml.as<std::string>();
          const auto maybe = findComponent(context, eid, component_name.c_str());
          if (!maybe) {
            GXF_LOG_ERROR("Could not find component with name '%s' in entity.",
                          component_name.c_str());
            return ForwardError(maybe);
          }
          cid = maybe.value();
          auto code = isSubgraph(context, cid);
          if (!code) {
            return Unexpected{GXF_FAILURE};
          }
          if (!code.value()) {
            // This is not a subgraph component, skip the loading
            Expected<YAML::Node> params = performParameterOverride(
              component, parameter_override, subgraph_entity_name);
            if (params) {
              all_components.insert({cid, params.value()});
            }
            continue;
          }
        } else {
          // Unexpected branch
          GXF_LOG_ERROR("If the type of a component is not specified it is interpreted as "
                        "setting parameters. In that case the component name is mandatory.");
          continue;
        }

        // This is a subgraph component, stash its parameters and starts the loading
        GXF_ASSERT_NE(cid, kNullUid);
        Expected<YAML::Node> params = performParameterOverride(
          component, parameter_override, subgraph_entity_name);
        if (params) {
          all_components.insert({cid, params.value()});
        }

        /* read the subgraph parameters */
        const char* key_location = "location";
        const char* key_prerequisites = "prerequisites";
        std::string yaml_sub;
        YAML::Node yaml_prerequisites;
        const auto component_parameters = component[kAttributeParameters];
        for (const auto& p : component_parameters) {
          const std::string key = p.first.as<std::string>();
          if (key == key_location) {
            yaml_sub = p.second.as<std::string>();
          } else if (key == key_prerequisites) {
            yaml_prerequisites = p.second;
          }
        }
        /* load the subgraphs from yaml files recursively if it is not loaded yet */
        const char* location = nullptr;
        auto result = GxfParameterGetPath(context, cid, key_location, &location);
        if (result == GXF_PARAMETER_NOT_INITIALIZED) {
          if (!yaml_sub.empty()) {
            std::vector<std::string> file_names;
            if (!split(yaml_sub, ",", file_names)) {
              GXF_LOG_ERROR("Failed to parse location string: %s", yaml_sub.c_str());
              return Unexpected{GXF_FAILURE};
            }
            for (auto &file_name : file_names) {
              file_name = root_.empty() || file_name.at(0) == '/' ?
                  file_name : root_ + '/' + file_name;
              current_graph_file = file_name;
              GXF_LOG_INFO("Loading Subgraph from YAML file '%s'...", file_name.c_str());
              auto yaml_nodes = YAML::LoadAllFromFile(file_name);
              FixedVector<YAML::Node, kMaxEntities> subgraph_nodes;
              auto result = StdVectorToFixedVector(yaml_nodes, subgraph_nodes)
                  .and_then([&](){
                    const char* entity_name = nullptr;
                    if (GXF_SUCCESS != GxfEntityGetName(
                      context, eid, &entity_name
                      )
                    ) {
                      entity_name = std::to_string(eid).c_str();
                    }
                    return load(context, subgraph_nodes, entity_name, eid,
                                parameters_override_string,
                                num_overrides,
                                yaml_prerequisites);
                  });
              if (!result) { return ForwardError(result); }
            }
          }
        } else {
          GXF_LOG_WARNING(
            "Subgraph %s already loaded once", name_yaml.as<std::string>().c_str());
        }
      }
    }

    // populate the interfaces if there is any
    for (size_t i = 0; i < nodes.size(); i++) {
      const auto& node = nodes.at(i).value();
      if (node[kAttributeInterfaces]) {
        const auto& interfaces = node[kAttributeInterfaces];
        if (!interfaces.IsSequence() && !interfaces.IsNull()) {
          GXF_LOG_ERROR("Interfaces must be Sequence");
          return Unexpected{GXF_FAILURE};
        }
        for (auto &interface : interfaces) {
          if (!interface[kAttributeName] ||
              !interface[kAttributeTarget]) {
            GXF_LOG_ERROR("Incomplete interface definition");
            return Unexpected{GXF_FAILURE};
            }
          const std::string interface_name = interface[kAttributeName].as<std::string>();
          const std::string tag = interface[kAttributeTarget].as<std::string>();
          auto result = addComponentToInterface(
            context, parent_eid, entity_prefix, interface_name, tag);
          if (!result) {
            return gxf::ForwardError(result);
          }
        }
      }
    }

    // Populates parameters after instantiation
    for (auto &component : all_components) {
      if (const auto& yaml_param = component.second) {
        const auto code = setParameters(
          context, component.first, entity_prefix, yaml_param);
        if (!code) {
          return ForwardError(code);
        }
      }
    }

    // populate EntityGroups to context if there is any
    const auto code = populateEntityGroups(context, nodes, entity_prefix);
    if (!code) {
      return ForwardError(code);
    }
    return Success;
  } catch (YAML::Exception& e) {
    GXF_LOG_ERROR("Could not create entities from nodes of yaml file %s: error - %s",
                  current_graph_file.c_str(), e.what());

    return Unexpected{GXF_FAILURE};
  }
}

// Emit a component's parameter key and value pair to a YAML::Emitter object
template <typename T>
Expected<void> emitComponentParameter(YAML::Emitter& out,
                                      std::shared_ptr<ParameterStorage> param_storage,
                                      gxf_uid_t cid, gxf_parameter_info_t& param_info) {
  const char* param_key = param_info.key;
  const auto maybe_param_value = param_storage->get<T>(cid, param_key);
  if (!maybe_param_value) {
    if (param_info.flags == GXF_PARAMETER_FLAGS_OPTIONAL) {
      GXF_LOG_INFO(
          "Could not get value of parameter \"%s\" for component C%05zu. "
          "Skipping as parameter is optional",
          param_key, cid);
      return Success;
    }
    if (maybe_param_value.error() != GXF_PARAMETER_NOT_INITIALIZED) {
      GXF_LOG_ERROR("Could not get value of parameter \"%s\" for component C%05zu", param_key, cid);
      return gxf::ForwardError(maybe_param_value);
    }
    // It is expected to not emit the key value pair if the value is not already initialized
    return Success;
  }
  out << YAML::Key << param_key;
  out << YAML::Value << maybe_param_value.value();
  return Success;
}

// Emit a component's parameter key and value pair to a YAML::Emitter object
template <typename T>
Expected<void> emitComponentParameter(YAML::Emitter& out,
                                      std::shared_ptr<ParameterStorage> param_storage,
                                      gxf_uid_t cid, const char* param_key) {
  const auto maybe_param_value = param_storage->get<T>(cid, param_key);
  if (!maybe_param_value) {
    if (maybe_param_value.error() != GXF_PARAMETER_NOT_INITIALIZED) {
      GXF_LOG_ERROR("Could not get value of parameter \"%s\" for component C%05zu", param_key, cid);
      return gxf::ForwardError(maybe_param_value);
    }
    // It is expected to not emit the key value pair if the value is not already initialized
    return Success;
  }
  out << YAML::Key << param_key;
  out << YAML::Value << maybe_param_value.value();
  return Success;
}

// Emit a component's parameter key and value pair to a YAML::Emitter object
Expected<void> emitComponentParameterViaWrap(YAML::Emitter& out,
                                             std::shared_ptr<ParameterStorage> param_storage,
                                             gxf_uid_t cid, gxf_parameter_info_t& param_info) {
  auto maybe_node = param_storage->wrap(cid, param_info.key);
  if (maybe_node) {
    out << YAML::Key << param_info.key;
    out << YAML::Value << maybe_node.value();
    return Success;
  } else if (param_info.flags == GXF_PARAMETER_FLAGS_OPTIONAL) {
    GXF_LOG_INFO(
        "Could not get value of parameter \"%s\" for component C%05zu. "
        "Skipping as parameter is optional",
        param_info.key, cid);
    return Success;
  } else {
    GXF_LOG_ERROR("Failed to wrap parameter '%s'with error %s", param_info.key,
                  GxfResultStr(maybe_node.error()));
    return Unexpected{maybe_node.error()};
  }
}

Expected<void> YamlFileLoader::saveToFile(
    gxf_context_t context, const std::string& filename) {
  if (context == nullptr) {
    return Unexpected{GXF_CONTEXT_INVALID};
  }
  if (filename.empty()) {
    GXF_LOG_ERROR("File name for exporting graph was empty.");
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  gxf_uid_t entities[kMaxEntities];
  uint64_t num_entities = kMaxEntities;
  const gxf_result_t result_1 = GxfEntityFindAll(context, &num_entities, entities);
  if (result_1 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find all entities");
    return Unexpected{result_1};
  }

  YAML::Emitter out;
  for (uint64_t i=0; i < num_entities; i++) {
    out << YAML::BeginDoc;
    out << YAML::BeginMap;

    const gxf_uid_t eid = entities[i];
    const char *entity_name;
    const gxf_result_t result_2 = GxfEntityGetName(context, eid, &entity_name);
    if (result_2 != GXF_SUCCESS) {
      GXF_LOG_ERROR("Could not get name for the entity E%05zu", eid);
      return Unexpected{result_2};
    }
    if (std::strcmp(entity_name, "") != 0) {
      out << YAML::Key << kAttributeName;
      out << YAML::Value << entity_name;
    }

    // Start exporting an entity's components
    out << YAML::Key << kAttributeComponents;
    out << YAML::Value;
    out << YAML::BeginSeq;

    // Get an entity's components
    gxf_uid_t cids[kMaxComponents];
    uint64_t num_cids = kMaxComponents;
    const gxf_result_t result_3 = GxfComponentFindAll(context, eid, &num_cids, cids);
    if (result_3 != GXF_SUCCESS) {
      GXF_LOG_ERROR(
        "Could not find all components for the entity %s (E%05zu)",
        entity_name, eid);
      return Unexpected{result_3};
    }

    // Export each component
    for (uint64_t j=0; j < num_cids; j++) {
      gxf_uid_t cid = cids[j];

      out << YAML::BeginMap;

      // Component name
      const char *comp_name;
      const gxf_result_t result_4 = GxfComponentName(context, cid, &comp_name);
      if (result_4 != GXF_SUCCESS) {
        GXF_LOG_ERROR(
          "Could not get name for component C%05zu in entity %s (E%05zu)",
          cid, entity_name, eid);
        return Unexpected{result_4};
      }
      // A component's name can be empty. Export only if it's not empty.
      if (std::strcmp(comp_name, "") != 0) {
        out << YAML::Key << kAttributeName;
        out << YAML::Value << comp_name;
      }

      // Component type
      gxf_tid_t tid;
      const char *comp_type_name;
      const gxf_result_t result_5 = GxfComponentType(context, cid, &tid);
      if (result_5 != GXF_SUCCESS) {
        GXF_LOG_ERROR("Could not get type for component %s/%s (C%05zu)",
          entity_name, comp_name, cid);
        return Unexpected{result_5};
      }
      const gxf_result_t result_6 = GxfComponentTypeName(context, tid, &comp_type_name);
      if (result_6 != GXF_SUCCESS) {
        GXF_LOG_ERROR("Could not get name for component type %016lx%016lx",
          tid.hash1, tid.hash2);
        return Unexpected{result_6};
      }
      out << YAML::Key << kAttributeType;
      out << YAML::Value << comp_type_name;

      // Component parameters
      out << YAML::Key << kAttributeParameters;
      out << YAML::Value;
      out << YAML::BeginMap;

      gxf_component_info_t comp_info;
      const char *parameter_name_ptrs[kMaxParameters];
      comp_info.parameters = parameter_name_ptrs;
      comp_info.num_parameters = kMaxParameters;
      const gxf_result_t result_7 = GxfComponentInfo(context, tid, &comp_info);
      if (result_7 != GXF_SUCCESS) {
        GXF_LOG_ERROR("Could not get info for component type %016lx%016lx",
          tid.hash1, tid.hash2);
        return Unexpected{result_7};
      }

      // Export each parameter
      for (uint64_t parameter_i=0; parameter_i < comp_info.num_parameters; parameter_i++) {
        gxf_parameter_info_t param_info;
        const gxf_result_t result_8 = GxfGetParameterInfo(
          context, tid, comp_info.parameters[parameter_i], &param_info);
        if (result_8 != GXF_SUCCESS) {
          GXF_LOG_ERROR("Could not get parameter info for component type %016lx%016lx",
            tid.hash1, tid.hash2);
          return Unexpected{result_8};
        }

        Expected<void> emit_result;
        switch (param_info.type) {
        case GXF_PARAMETER_TYPE_HANDLE:
        {
          auto maybe_handle = parameter_storage_->getHandle(cid, param_info.key);
          if (!maybe_handle) {
              if (param_info.flags == GXF_PARAMETER_FLAGS_OPTIONAL) {
                GXF_LOG_DEBUG(
                    "Could not get parameter \"%s\" as handle for component %s/%s (C%05zu). "
                    "Skipping as param is optional",
                    param_info.key, entity_name, comp_name, cid);
                continue;
              }
            GXF_LOG_ERROR(
              "Could not get parameter \"%s\" as handle for component %s/%s (C%05zu) %s",
              param_info.key, entity_name, comp_name, cid, GxfResultStr(maybe_handle.error()));
            emit_result = gxf::ForwardError(maybe_handle);
            break;
          }
          const char *handle_comp_name;
          const auto result_9 = GxfComponentName(context, maybe_handle.value(), &handle_comp_name);
          if (result_9 != GXF_SUCCESS) {
            GXF_LOG_ERROR(
              "Could not get name of handle C%05zu of parameter \"%s\""
                " for component %s/%s (C%05zu)",
              maybe_handle.value(), param_info.key, entity_name, comp_name, cid);
            emit_result = Unexpected{result_9};
            break;
          }
          out << YAML::Key << param_info.key;
          out << YAML::Value << handle_comp_name;
          emit_result = Success;
          break;
        }
        case GXF_PARAMETER_TYPE_STRING:
        {
          emit_result = emitComponentParameter<const char*>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_INT8:
        {
          emit_result = emitComponentParameter<int8_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_INT16:
        {
          emit_result = emitComponentParameter<int16_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_INT32:
        {
          emit_result = emitComponentParameter<int32_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_INT64:
        {
          emit_result = emitComponentParameter<int64_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_UINT8:
        {
          emit_result = emitComponentParameter<uint8_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_UINT16:
        {
          emit_result = emitComponentParameter<uint16_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_UINT32:
        {
          emit_result = emitComponentParameter<uint32_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_UINT64:
        {
          emit_result = emitComponentParameter<uint64_t>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_FLOAT32:
        {
          emit_result = emitComponentParameter<float>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_FLOAT64:
        {
          emit_result = emitComponentParameter<double>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_COMPLEX64:
        {
          emit_result = emitComponentParameter<std::complex<float>>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_COMPLEX128:
        {
          emit_result = emitComponentParameter<std::complex<double>>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_BOOL:
        {
          emit_result = emitComponentParameter<bool>(
            out, parameter_storage_, cid, param_info);
          break;
        }
        case GXF_PARAMETER_TYPE_FILE:
        {
          emit_result = emitComponentParameterViaWrap(
            out, parameter_storage_, cid, param_info);
            break;
        }
        case GXF_PARAMETER_TYPE_CUSTOM: {
          emit_result = emitComponentParameterViaWrap(
            out, parameter_storage_, cid, param_info);
          break;
        }
        default: {
          GXF_LOG_ERROR("Parameter type %s was not supported for exporting",
                        GxfParameterTypeStr(param_info.type));
          emit_result = Unexpected{GXF_NOT_IMPLEMENTED};
        }
        }

        if (!emit_result) {
          GXF_LOG_ERROR(
            "Failed to emit parameter \"%s\" for component %s/%s (C%05zu)",
            param_info.key, entity_name, comp_name, cid);
          return gxf::ForwardError(emit_result);
        }
      }
      // Close map for a component's parameters
      out << YAML::EndMap;
      // Close map for the component
      out << YAML::EndMap;
    }
    // Close seq for the component list in an entity
    out << YAML::EndSeq;
    // Close map for an entity
    out << YAML::EndMap;
  }

  std::ofstream myfile(filename);
  if (!myfile.is_open()) {
    GXF_LOG_ERROR("Could not open file \"%s\" for exporting graph", filename.c_str());
    return Unexpected{GXF_FAILURE};
  }
  myfile << out.c_str();
  myfile << "\n";
  myfile.close();
  GXF_LOG_INFO("Successfully exported graph to \"%s\"", filename.c_str());
  return Success;
}

Expected<gxf_uid_t> YamlFileLoader::findOrCreateEntity(
    gxf_context_t context, const Expected<std::string>& entity_name) {
  // Was a name given?
  // no: create new entity and return it
  // yes: try to find entity
  //  found: return link to existing entity
  //  not found: create new entity with that name and return it

  // try to find an existing entity
  if (entity_name && !entity_name->empty()) {
    gxf_uid_t eid;
    const gxf_result_t result_1 = GxfEntityFind(context, entity_name->c_str(), &eid);
    if (result_1 == GXF_SUCCESS) { return eid; }
    if (result_1 != GXF_ENTITY_NOT_FOUND) { return Unexpected{result_1}; }
  }

  // create a new entity
  const GxfEntityCreateInfo info{
      entity_name ? entity_name->c_str() : nullptr,
      GXF_ENTITY_CREATE_PROGRAM_BIT
  };
  gxf_uid_t eid;
  const gxf_result_t result_2 = GxfCreateEntity(context, &info, &eid);
  if (result_2 != GXF_SUCCESS) { return Unexpected{result_2}; }

  return eid;
}

Expected<gxf_uid_t> YamlFileLoader::addComponent(gxf_context_t context, gxf_uid_t eid,
                                                 const char* type) {
  gxf_tid_t tid;
  const auto result_1 = GxfComponentTypeId(context, type, &tid);
  if (result_1 != GXF_SUCCESS) return Unexpected{result_1};
  gxf_uid_t cid;
  const auto result_2 = GxfComponentAdd(context, eid, tid, nullptr, &cid);
  if (result_2 != GXF_SUCCESS) return Unexpected{result_2};
  return cid;
}

Expected<gxf_uid_t> YamlFileLoader::findComponent(gxf_context_t context, gxf_uid_t eid,
                                                  const char* name) {
  int32_t offset = 0;
  gxf_uid_t cid;
  const auto result_1 = GxfComponentFind(context, eid, GxfTidNull(), name, &offset, &cid);
  if (result_1 != GXF_SUCCESS) { return Unexpected{result_1}; }
  // Look again to make sure it's unique
  offset++;
  const auto result_2 = GxfComponentFind(context, eid, GxfTidNull(), name, &offset, &cid);
  if (result_2 == GXF_SUCCESS) { return Unexpected{GXF_FAILURE}; }
  if (result_2 != GXF_ENTITY_COMPONENT_NOT_FOUND) { return Unexpected{result_2}; }
  return cid;
}

Expected<void> YamlFileLoader::setParameters(gxf_context_t context, gxf_uid_t handle,
    const std::string& prefix, const YAML::Node& parameters) {
  if (!parameters.IsMap()) {
    GXF_LOG_ERROR("Could not parse parameters for %s- not a map", prefix.c_str());
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  if (parameter_storage_ == nullptr) { return Unexpected{GXF_NULL_POINTER}; }

  for (const auto& p : parameters) {
    const std::string key = p.first.as<std::string>();

    // Parse the parameter as specified in the backend
    const Expected<void> result = parameter_storage_->parse(handle, key.c_str(), p.second, prefix);
    if (result) {
      continue;
    }

    // If the parameter was not found in the parameter interface we still continue parsing to
    // support dynamic, unregistered parameters.
    if (result.error() != GXF_PARAMETER_NOT_FOUND) {
      return ForwardError(result);
    }
    const char* component_name = "UNKNOWN";
    GxfParameterGetStr(context, handle, kInternalNameParameterKey, &component_name);
    GXF_LOG_WARNING("Using unregistered parameter '%s' in component '%s'.", key.c_str(),
                    component_name);
    if (p.second.IsScalar()) {
      // bool
      try {
        const bool value = p.second.as<bool>();
        const gxf_result_t code = GxfParameterSetBool(context, handle, key.c_str(), value);
        if (code != GXF_SUCCESS) {
          return Unexpected{code};
        }
        continue;
      } catch (...) {}
      // int64_t
      try {
        int64_t value = p.second.as<int64_t>();
        if (p.second.Tag() == "!ms") {
          value *= 1'000'000;
        } else {
          if (p.second.Tag() != "?") {
            GXF_LOG_ERROR("Unknown tag '%s'", p.second.Tag().c_str());
            return Unexpected{GXF_INVALID_DATA_FORMAT};
          }
        }
        const gxf_result_t code = GxfParameterSetInt64(context, handle, key.c_str(), value);
        if (code != GXF_SUCCESS) {
          return Unexpected{code};
        }
        continue;
      } catch (...) {}
      // float64
      try {
        const double value = p.second.as<double>();
        const gxf_result_t code = GxfParameterSetFloat64(context, handle, key.c_str(), value);
        if (code != GXF_SUCCESS) {
          return Unexpected{code};
        }
        continue;
      } catch (...) {}
        // string by default
        const gxf_result_t code = GxfParameterSetStr(context, handle, key.c_str(),
                                                     p.second.as<std::string>().c_str());
        if (code != GXF_SUCCESS) {
          return Unexpected{code};
        }
    } else {
      if (p.second.Tag() == "!yaml") {
        std::stringstream ss;
        ss << p.second;
        const gxf_result_t code = GxfParameterSetStr(context, handle, key.c_str(),
                                                     ss.str().c_str());
        if (code != GXF_SUCCESS) {
          return Unexpected{code};
        }
      } else {
        if (p.second.IsSequence()) {
          for (size_t i = 0; i < p.second.size(); i++) {
            const auto code = setParameters(context, handle,
                                            (key + "-" + std::to_string(i) + "/").c_str(),
                                            p.second[i]);
            if (!code) {
              return code;
            }
          }
        } else if (p.second.IsMap()) {
          const auto code = setParameters(context, handle, (key + "/").c_str(), p.second);
          if (!code) {
            return code;
          }
        } else {
          GXF_LOG_ERROR("unsupported value type");
        }
      }
    }
  }
  return Success;
}

Expected<void> YamlFileLoader::addComponentToInterface(
    gxf_context_t context, gxf_uid_t eid, const std::string& entity_prefix,
    const std::string& interface_name, const std::string& tag) {
  gxf_uid_t target_eid;
  std::string target_component_name;
  const size_t pos = tag.find('/');
  if (pos == std::string::npos) {
    GXF_LOG_ERROR("Incomplete target for interface or prerequisites mapping");
    return Unexpected{GXF_FAILURE};
  }
  // Split the tag into entity and component name
  const std::string target_entity_name = entity_prefix.empty() ?
                            tag.substr(0, pos) : (entity_prefix + tag.substr(0, pos));
  target_component_name = tag.substr(pos + 1);

  // Search for the entity
  const gxf_result_t result_1 = GxfEntityFind(context, target_entity_name.c_str(), &target_eid);
  if (result_1 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find entity '%s'", target_entity_name.c_str());
    return Unexpected{result_1};
  }
  gxf_uid_t target_cid;
  const gxf_result_t result_2 = GxfComponentFind(context, target_eid, GxfTidNull(),
                                                 target_component_name.c_str(), nullptr,
                                                 &target_cid);
  if (result_2 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component '%s' in entity 'name:%s - id:%zu'",
                  target_component_name.c_str(), target_entity_name.c_str(), target_eid);
    return Unexpected{result_2};
  }

  const gxf_result_t result_3 = GxfComponentAddToInterface(context, eid, target_cid,
                                                            interface_name.c_str());
  if (result_3 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Couldn't add component %s to the interface of %s",
                    target_component_name.c_str(), target_entity_name.c_str());
    return Unexpected{result_3};
  }

  return Success;
}

Expected<void> YamlFileLoader::populateEntityGroups(gxf_context_t context,
                                                    const FixedVectorBase<YAML::Node>& nodes,
                                                    const std::string& entity_prefix) {
  for (size_t i = 0; i < nodes.size(); i++) {
    const auto& node = nodes.at(i).value();
    if (node[kAttributeEntityGroups]) {
      const auto& entity_groups = node[kAttributeEntityGroups];
      if (!entity_groups.IsSequence() && !entity_groups.IsNull()) {
        GXF_LOG_ERROR("EntityGroups must be Sequence");
        return Unexpected{GXF_FAILURE};
      }
      for (auto &entity_group_item : entity_groups) {
        if (!entity_group_item[kAttributeName] ||
            !entity_group_item[kAttributeTarget]) {
          GXF_LOG_ERROR("Incomplete EnityGroup definition");
          return Unexpected{GXF_FAILURE};
        }
        const std::string entity_group_name = entity_group_item[kAttributeName].as<std::string>();
        const std::vector<std::string> entity_list =
          entity_group_item[kAttributeTarget].as<std::vector<std::string>>();
        // Create empty EntityGroup
        gxf_uid_t gid = kNullUid;
        gxf_result_t result = GxfCreateEntityGroup(context, entity_group_name.c_str(), &gid);
        if (result != GXF_SUCCESS) {
          return Unexpected{result};
        }
        for (std::string entity_name : entity_list) {
          gxf_uid_t eid = kNullUid;
          if (!entity_prefix.empty()) {
            entity_name = entity_prefix + entity_name;
          }
          gxf_result_t result = GxfEntityFind(context, entity_name.c_str(), &eid);
          if (GXF_SUCCESS != result) {
            return Unexpected{result};
          }
          // Add each Entity into just created EntityGroup
          result = GxfUpdateEntityGroup(context, gid, eid);
          if (result != GXF_SUCCESS) {
            GXF_LOG_ERROR("Failed to add entity [name: %s, eid: %ld] to EntityGroup %s",
                          entity_name.c_str(), eid, entity_group_name.c_str());
            return Unexpected{result};
          }
        }
      }
    }
  }
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
