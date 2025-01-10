/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/graph_driver_worker_common.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

#include "gxf/core/parameter_registrar.hpp"

namespace nvidia {
namespace gxf {

void parseIpAddress(const std::string& ip_address_port, std::string& ip_address, int& port) {
  if (ip_address_port.empty()) {
    GXF_LOG_DEBUG("Empty ip_address_port to break");
    return;
  }
  int pivot = ip_address_port.find(":");
  ip_address = ip_address_port.substr(0, pivot);
  port = std::stoi(ip_address_port.substr(pivot + 1));
  return;
}

void parseSegmentEntityComponentName(
  const std::string& segment_entity_component_name,
  std::string& segment_name,
  std::string& entity_name,
  std::string& component_name
) {
  int first_pivot = segment_entity_component_name.find(".");
  int second_pivot = segment_entity_component_name.substr(first_pivot + 1).find(".");
  int third_pivot = segment_entity_component_name.find_last_of(".");
  segment_name = segment_entity_component_name.substr(0, first_pivot);
  entity_name = segment_entity_component_name.substr(first_pivot + 1, second_pivot);
  component_name = segment_entity_component_name.substr(third_pivot + 1);
  return;
}

// payload JSON Schema
const char* jsonSchemaOnSetComponentParams = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "pattern": "^.*/.*/.*$"
      },
      "params": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "key": {
              "type": "string"
            },
            "value": {
              "type": "string"
            },
            "value_type": {
              "type": "string"
            }
          },
          "required": ["key", "value", "value_type"]
        }
      }
    },
    "required": ["name", "params"]
  }
})json";
/** ----------------------------------------------------
Example Json instance:
[
  {
    "name": "SegmentName1/EntityName1/ComponentName1",
    "params": [
      {
        "key": "ParamKey1",
        "value": "ParamValue1",
        "value_type": "String"
      },
      {
        "key": "ParamKey2",
        "value": "ParamValue2",
        "value_type": "Int32"
      }
      // ... other ParamInfo objects
    ]
  },
  {
    "name": "SegmentName2/EntityName2/ComponentName2",
    "params": [
      {
        "key": "ParamKey3",
        "value": "ParamValue3",
        "value_type": "Bool"
      }
      // ... other ParamInfo objects
    ]
  }
  // ... other ComponentParam objects
]
*/
Expected<std::vector<ComponentParam>>
GraphDriverWorkerParser::deserialize_onSetComponentParams(const std::string& payload) {
  bool validParamInfo = true;
  std::stringstream errorStream;
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(payload);
  } catch (const nlohmann::json::parse_error& e) {
    GXF_LOG_ERROR("Failed to parse json file %s with error %s", payload.c_str(), e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  std::vector<ComponentParam> component_param_list;

  for (const auto& item : json) {
    ComponentParam component_param;
    // Check for "name" key
    if (item.find("name") == item.end()) {
      errorStream << "Error: Missing 'name' key in JSON object.\n";
    } else if (!item["name"].is_string()) {
      errorStream << "Error: 'name' key is not a string in JSON object.\n";
    } else {
      std::string name = item["name"];
      std::stringstream nameStream(name);
      std::string segment_name;
      std::getline(nameStream, segment_name, '.');
      std::string entity_name;
      std::getline(nameStream, entity_name, '.');
      std::string component_name;
      std::getline(nameStream, component_name, '.');
      if (!segment_name.empty() && !entity_name.empty() && !component_name.empty()) {
        component_param.segment_name = segment_name;
        component_param.entity_name = entity_name;
        component_param.component_name = component_name;
      } else {
        errorStream << "Error: 'name' key does not have the expected format "
          "'SegmentName.EntityName.ComponentName'.\n";
        if (segment_name.empty()) {
          errorStream << " Missing 'SegmentName'.";
        }
        if (entity_name.empty()) {
          errorStream << " Missing 'EntityName'.";
        }
        if (component_name.empty()) {
          errorStream << " Missing 'ComponentName'.";
        }
      }
    }

    // Check for "params" key and its contents
    if (!item.contains("params") || !item["params"].is_array()) {
      errorStream << "Error: Missing or invalid 'params' key in JSON object.\n";
    } else {
      for (const auto& param_item : item["params"]) {
        ComponentParam::ParamInfo paramInfo;
        // key
        if (!param_item.contains("key")) {
          errorStream << "Error: Missing 'key' in 'params' array.\n";
          validParamInfo = false;
        } else if (!param_item["key"].is_string()) {
          errorStream << "Error: 'key' is not a string in 'params' array.\n";
          validParamInfo = false;
        } else {
          paramInfo.key = param_item["key"];
        }
        // value
        if (!param_item.contains("value")) {
          errorStream << "Error: Missing 'value' in 'params' array.\n";
          validParamInfo = false;
        } else if (!param_item["value"].is_string()) {
          errorStream << "Error: 'value' is not a string in 'params' array.\n";
          validParamInfo = false;
        } else {
          paramInfo.value = param_item["value"];
        }
        // value_type
        if (!param_item.contains("value_type")) {
          errorStream << "Error: Missing 'value_type' in 'params' array.\n";
          validParamInfo = false;
        } else if (!param_item["value_type"].is_string()) {
          errorStream << "Error: 'value_type' is not a string in 'params' array.\n";
          validParamInfo = false;
        } else {
          if (validParamInfo) {
            paramInfo.value_type = param_item["value_type"];
            if (paramInfo.value_type == ParameterTypeTrait<bool>::type_name) {
              auto value = ComponentParam::ParamInfo::strToBool(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            } else if (paramInfo.value_type == ParameterTypeTrait<float>::type_name) {
              auto value = ComponentParam::ParamInfo::strToFloat32(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            } else if (paramInfo.value_type == ParameterTypeTrait<double>::type_name) {
              auto value = ComponentParam::ParamInfo::strToFloat64(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            } else if (paramInfo.value_type == ParameterTypeTrait<uint16_t>::type_name) {
              auto value = ComponentParam::ParamInfo::strToUInt16(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            } else if (paramInfo.value_type == ParameterTypeTrait<int32_t>::type_name) {
              auto value = ComponentParam::ParamInfo::strToInt32(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            } else if (paramInfo.value_type == ParameterTypeTrait<int64_t>::type_name) {
              auto value = ComponentParam::ParamInfo::strToInt64(paramInfo.value);
              if (!value) { validParamInfo = false; }
            } else if (paramInfo.value_type == ParameterTypeTrait<uint32_t>::type_name) {
              auto value = ComponentParam::ParamInfo::strToUInt32(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            } else if (paramInfo.value_type == ParameterTypeTrait<uint64_t>::type_name) {
              auto value = ComponentParam::ParamInfo::strToUInt64(paramInfo.value);
              if (!value) {
                validParamInfo = false;
                errorStream << "Error: " << paramInfo.value <<
                  " is not a valid " << paramInfo.value_type << ".\n";
              }
            }
          }
        }
        // complete one paramInfo
        if (validParamInfo) {
          component_param.params.push_back(paramInfo);
        }
      }
    }

    // If there are any errors, output them all at once
    if (!errorStream.str().empty()) {
      GXF_LOG_ERROR("%s", errorStream.str().c_str());
    } else {  // If no errors, add param to params vector
      component_param_list.push_back(component_param);
    }
  }

  if (!errorStream.str().empty()) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return component_param_list;
}

Expected<std::string> GraphDriverWorkerParser::serialize_onSetComponentParams(
  const std::vector<ComponentParam>& component_param_list
) {
  nlohmann::json json_array = nlohmann::json::array();

  for (const auto& component : component_param_list) {
    nlohmann::json json_object;
    json_object["name"] = component.segment_name + "." +
      component.entity_name + "." + component.component_name;

    // Create JSON array for params
    nlohmann::json params_array = nlohmann::json::array();
    for (const auto& param : component.params) {
      nlohmann::json param_object;
      param_object["key"] = param.key;
      param_object["value"] = param.value;
      param_object["value_type"] = param.value_type;
      params_array.push_back(param_object);
    }
    json_object["params"] = params_array;
    json_array.push_back(json_object);  // Add component object to array
  }

  return json_array.dump();  // Serialize the JSON array to a string
}

// payload JSON Schema
const char* jsonSchemaOnRegisterGraphWorker = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "WorkerInfo",
  "type": "object",
  "properties": {
    "server_ip_address": {
      "type": "string",
      "format": "ipv4"
    },
    "server_port": {
      "type": "string",
      "pattern": "^[0-9]+$"
    },
    "segment_info_list": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "segment_name": {
            "type": "string"
          },
          "ip_port_address_map": {
            "type": "object",
            "additionalProperties": {
              "type": "string",
              "format": "uri"
            }
          }
        },
        "required": ["segment_name", "ip_port_address_map"]
      }
    }
  },
  "required": ["server_ip_address", "server_port", "segment_info_list"]
})json";
/** ----------------------------------------------------
Example Json instance:
{
  "server_ip_address": "192.168.1.10",
  "server_port": "8080",
  "segment_info_list": [
    {
      "segment_name": "SegmentA/Entity1/UcRx1",
      "ip_port_address_map": {
        "SegmentA/Entity1/UcRx1": "10.0.0.1:3000",
        "SegmentA/Entity2/UcRx2": "10.0.0.2:3001"
      }
    },
    {
      "segment_name": "SegmentB/Entity3/UcRx3",
      "ip_port_address_map": {
        "SegmentB/Entity3/UcRx3": "10.0.0.3:3002",
        "SegmentB/Entity4/UcRx4": "10.0.0.4:3003"
      }
    }
  ]
}
*/
Expected<WorkerInfo> GraphDriverWorkerParser::deserialize_onRegisterGraphWorker(
  const std::string& payload
) {
  std::stringstream errorStream;
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(payload);
  } catch (const nlohmann::json::parse_error& e) {
    GXF_LOG_ERROR("Failed to parse json file %s with error %s", payload.c_str(), e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  WorkerInfo worker_info;

  // Deserialize server IP address and port
  if (!json.contains("server_ip_address") || !json["server_ip_address"].is_string()) {
    errorStream << "Error: Missing or invalid 'server_ip_address' in JSON payload.\n";
  } else {
    worker_info.server_ip_address = json["server_ip_address"];
  }
  if (!json.contains("server_port") || !json["server_port"].is_string()) {
    errorStream << "Error: Missing or invalid 'server_port' in JSON payload.\n";
  } else {
    worker_info.server_port = json["server_port"];
  }

  // Deserialize segment info list
  if (!json.contains("segment_info_list")) {
    // empty segment load, sanity test case
  } else if (!json["segment_info_list"].is_array()) {
    errorStream << "Error: Invalid 'segment_info_list' in JSON payload.\n";
  } else {
    for (const auto& seg_info_json : json["segment_info_list"]) {
      if (!seg_info_json.contains("segment_name") || !seg_info_json["segment_name"].is_string()) {
        errorStream << "Error: Missing or invalid 'segment_name' in segment info.\n";
      } else {
        SegmentInfo seg_info;
        seg_info.segment_name = seg_info_json["segment_name"];
        if (!seg_info_json.contains("ip_port_address_map") ||
          !seg_info_json["ip_port_address_map"].is_object()) {
          if (seg_info_json["ip_port_address_map"].is_null()) {
            GXF_LOG_DEBUG("Empty ip_port_address_map");
          } else {
            errorStream << "Error: Missing or invalid 'ip_port_address_map' in segment info.\n";
          }
        } else {
          for (const auto& pair : seg_info_json["ip_port_address_map"].items()) {
            if (!pair.value().is_string()) {
              errorStream << "Error: Invalid IP port address format for '" << pair.key() << "'.\n";
            } else {
              seg_info.ip_port_address_map[pair.key()] = pair.value();
            }
          }
        }
        worker_info.segment_info_list.push_back(seg_info);
      }
    }
  }

  // If there are any errors, output them all at once
  if (!errorStream.str().empty()) {
    GXF_LOG_ERROR("%s\npayload: %s", errorStream.str().c_str(), payload.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return worker_info;
}

Expected<std::string> GraphDriverWorkerParser::serialize_onRegisterGraphWorker(
  const WorkerInfo& worker_info
) {
  nlohmann::json json;

  // Serialize server IP address and port
  json["server_ip_address"] = worker_info.server_ip_address;
  json["server_port"] = worker_info.server_port;

  // Serialize segment info list
  for (const auto& seg_info : worker_info.segment_info_list) {
    nlohmann::json seg_info_json;
    seg_info_json["segment_name"] = seg_info.segment_name;

    nlohmann::json ip_port_map_json;
    for (const auto& ip_port_pair : seg_info.ip_port_address_map) {
      ip_port_map_json[ip_port_pair.first] = ip_port_pair.second;
    }
    seg_info_json["ip_port_address_map"] = ip_port_map_json;
    json["segment_info_list"].push_back(seg_info_json);
  }

  return json.dump();  // Serialize the JSON object to a string
}

//
// helpers
//
Expected<bool> ComponentParam::ParamInfo::strToBool(const std::string& str) {
  std::string lowerStr = str;
  std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
  return lowerStr == "true" || lowerStr == "1";
}
Expected<int32_t> ComponentParam::ParamInfo::strToInt32(const std::string& str) {
  try {
    int32_t value_int32 = std::stoi(str);
    return value_int32;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}
Expected<uint32_t> ComponentParam::ParamInfo::strToUInt32(const std::string& str) {
  try {
    uint32_t value_uint32 = static_cast<uint32_t>(std::stoul(str));
    return value_uint32;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}
Expected<int64_t> ComponentParam::ParamInfo::strToInt64(const std::string& str) {
  try {
    int64_t value_int64 = std::stoll(str);
    return value_int64;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}
Expected<uint64_t> ComponentParam::ParamInfo::strToUInt64(const std::string& str) {
  try {
    uint64_t value_uint64 = std::stoull(str);
    return value_uint64;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}
Expected<float> ComponentParam::ParamInfo::strToFloat32(const std::string& str) {
  try {
    float value_float = std::stof(str);
    return value_float;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}
Expected<double> ComponentParam::ParamInfo::strToFloat64(const std::string& str) {
  try {
    double value_double = std::stod(str);
    return value_double;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}
Expected<uint16_t> ComponentParam::ParamInfo::strToUInt16(const std::string& str) {
  try {
    uint16_t value_uint16 = static_cast<uint16_t>(std::stoul(str));
    return value_uint16;
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("Exception: %s", e.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
}

}  // namespace gxf
}  // namespace nvidia
