/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/app/config_parser.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nvidia {
namespace gxf {

Expected<std::shared_ptr<ConfigParser::SegmentConfig>> ConfigParser::getSegmentConfig() {
  if (file_path_.empty()) {
    GXF_LOG_INFO("No config file provided, return default segment control");
    return segment_control_;
  }
  std::vector<YAML::Node> yaml_nodes;
  // read yaml file first
  try {
    yaml_nodes = YAML::LoadAllFromFile(file_path_);
  } catch (const std::exception& exception) {
    GXF_LOG_ERROR("Failed to read config yaml file: %s, exception: %s",
      file_path_.c_str(), exception.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  } catch (...) {
    GXF_LOG_ERROR("Failed to read config yaml file: %s for unknown reason",
    file_path_.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // parse yaml file
  try {
    for (size_t i = 0; i < yaml_nodes.size(); i++) {
      const auto& node = yaml_nodes.at(i);
      if (node[kSegmentConfig]) {
        // Get the entity name
        std::string entity_name;
        if (const auto& name_yaml = node[kName]) {
          entity_name = name_yaml.as<std::string>();
        }

        const auto& members = node[kSegmentConfig];
        if (!members.IsSequence() && !members.IsNull()) {
          GXF_LOG_ERROR("members must be Sequence");
          return Unexpected{GXF_INVALID_DATA_FORMAT};
        }
        if (members.size() == 0) {
          GXF_LOG_WARNING("Segment control not configured, will use default behavior");
        }

        for (auto member : members) {
          if (!member[kMemberType]) {
            GXF_LOG_ERROR("Each segment control section need a 'member' key value pair");
            return Unexpected{GXF_INVALID_DATA_FORMAT};
          }
          if (!member[kMemberParam]) {
            GXF_LOG_ERROR("Each segment control section need a 'parameter' key value pair");
            return Unexpected{GXF_INVALID_DATA_FORMAT};
          }
          const auto& member_type_yaml = member[kMemberType];
          const std::string member_type_name = member_type_yaml.as<std::string>();
          const auto& member_parameters_yaml = member[kMemberParam];
          if (member_type_name == kEnabledSegments) {
            segment_control_->enable_all_segments = false;
            for (const auto& p : member_parameters_yaml) {
              const std::string key = p.first.as<std::string>();
              if (key == kEnabled) {
                segment_control_->enabled_segments.names = p.second.as<std::vector<std::string>>();
              }
            }
            if (segment_control_->enabled_segments.names.empty()) {
              GXF_LOG_ERROR("Empty enabled segment list");
            }
          } else if (member_type_name == kWorker) {
            for (const auto& p : member_parameters_yaml) {
              const std::string key = p.first.as<std::string>();
              if (key == kEnabled) {
                segment_control_->worker.enabled = p.second.as<bool>();
              } else if (key == kName) {
                segment_control_->worker.name = p.second.as<std::string>();
              } else if (key == kPort) {
                segment_control_->worker.port = p.second.as<int32_t>();
              } else if (key == kDriverIp) {
                segment_control_->worker.driver_ip = p.second.as<std::string>();
              } else if (key == kDriverPort) {
                segment_control_->worker.driver_port = p.second.as<int32_t>();
              } else {
                GXF_LOG_ERROR("Invalid SegmentConfig Worker parameter: %s", key.c_str());
                return Unexpected{GXF_INVALID_DATA_FORMAT};
              }
            }
          } else if (member_type_name == kDriver) {
            for (const auto& p : member_parameters_yaml) {
              const std::string key = p.first.as<std::string>();
              if (key == kEnabled) {
                segment_control_->driver.enabled = p.second.as<bool>();
              } else if (key == kName) {
                segment_control_->driver.name = p.second.as<std::string>();
              } else if (key == kPort) {
                segment_control_->driver.port = p.second.as<int32_t>();
              } else {
                GXF_LOG_ERROR("Invalid SegmentConfig Driver parameter: %s", key.c_str());
                return Unexpected{GXF_INVALID_DATA_FORMAT};
              }
            }
          } else {
            GXF_LOG_ERROR("Invalid SegmentConfig member: %s", member_type_name.c_str());
            return Unexpected{GXF_INVALID_DATA_FORMAT};
          }
        }
      }
    }
  } catch (const std::exception& exception) {
    GXF_LOG_ERROR("Failed to parse config yaml file: %s, exception: %s",
      file_path_.c_str(), exception.what());
    return Unexpected{GXF_ARGUMENT_INVALID};
  } catch (...) {
    GXF_LOG_ERROR("Failed to parse config yaml file: %s for unknown reason",
      file_path_.c_str());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  return segment_control_;
}

Expected<void> ConfigParser::setFilePath(const std::string& file_path) {
  if (!file_path_.empty()) {
    GXF_LOG_INFO("Changing config file %s -> %s",
      file_path_.c_str(), file_path.c_str());
  }
  file_path_ = file_path;
  return Success;
}

Expected<void> ConfigParser::setFilePath(int argc, char** argv) {
  if (argc > 1) {
    std::string file_path = argv[1];
    if (std::filesystem::path(file_path).is_absolute()) {
      if (std::filesystem::exists(file_path)) {
        file_path_ = file_path;
        return Success;
      } else {
        GXF_LOG_ERROR("Config file no found at: %s", file_path.c_str());
        return Unexpected{GXF_ARGUMENT_INVALID};
      }
    } else {
      file_path_ = getExecutablePath() + "/" + file_path;
      if (!std::filesystem::exists(file_path)) {
        GXF_LOG_ERROR("Config file no found at: %s", file_path.c_str());
        return Unexpected{GXF_ARGUMENT_INVALID};
      }
      return Success;
    }
  } else {
    GXF_LOG_INFO("Seg config file API called without providing config file at runtime");
  }
  return Success;
}

std::string ConfigParser::getExecutablePath() {
#if defined(__linux__)
  return std::filesystem::canonical("/proc/self/exe").parent_path().string();
#else
  return std::filesystem::canonical(std::filesystem::path(__FILE__).parent_path()).string();
#endif
}

}  // namespace gxf
}  // namespace nvidia
