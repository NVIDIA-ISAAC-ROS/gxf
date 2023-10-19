/*
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>

#include <thread>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"

#define MAX_EXT_COUNT 100
#define MAX_COMP_COUNT 300
#define MAX_PARAM_COUNT 500

gxf_result_t GxfComponentAddByTypeName(gxf_context_t context, gxf_uid_t eid,
                                       const char* component_type_name, const char* component_name,
                                       gxf_uid_t* cid) {
  gxf_tid_t tid;
  gxf_result_t result = GxfComponentTypeId(context, component_type_name, &tid);
  if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  result = GxfComponentAdd(context, eid, tid, component_name, cid);
  if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  return result;
}

/**
 * @brief Gives a new runtime info objects with memory allocated
 *        to fetch extension tids
 *
 * @param count Expected number of extensions in the runtime
 * @return gxf_runtime_info
 */
gxf_runtime_info new_runtime_info(uint64_t count = 0) {
  count = count ? count : MAX_EXT_COUNT;
  gxf_runtime_info info;
  info.extensions = new gxf_tid_t[count];
  info.num_extensions = count;
  return info;
}

/**
 * @brief Free dyamic memory allocated in runtime info
 *
 * @param info gxf_runtime_info
 */
void destruct_runtime_info(gxf_runtime_info& info) {
  info.version = nullptr;
  info.num_extensions = 0;
  delete[] info.extensions;
}

/**
 * @brief Resize the dynamic memory allocated in runtime info
 *
 * @param info gxf_runtime_info to be resized
 * @param count new number of expected extensions
 */
void realloc_runtime_info(gxf_runtime_info& info, uint64_t count) {
  destruct_runtime_info(info);
  info = new_runtime_info(count);
}

/**
 * @brief Converts a decimal number to a hex string of length 16
 *
 * @param dec Decimal number (example : 6656714950019203559)
 * @return std::string hex number (example : 5c6166fa6eed41e7)
 */
inline std::string uint64_to_hex_string(uint64_t dec) {
  std::ostringstream oss;
  oss << std::hex << dec;
  std::string result = oss.str();
  while (result.length() < 16) result = "0" + result;
  return result;
}

/**
 * @brief Converts a hex string to a uint64_t decimal number
 *
 * @param hex string containing a hex number (example : 5c6166fa6eed41e7)
 * @return uint64_t Decimal number (example : 6656714950019203559)
 */
inline uint64_t hex_string_to_uint64(std::string hex) {
  uint64_t result;
  std::istringstream iss(hex);
  iss >> std::hex >> result;
  return result;
}

/**
 * @brief Converts a gxf_tid to a uuid string
 *
 * @param tid A valid tid from gxf runtime
 * @return std::string UUID of type 85f64c84-8236-4035-9b9a-3843a6a2026f
 */
inline std::string tid_to_uuid(gxf_tid_t& tid) {
  std::string raw(uint64_to_hex_string(tid.hash1) + uint64_to_hex_string(tid.hash2));
  if (raw.length() != 32) GXF_LOG_ERROR("Invalid tid : %s ", raw.c_str());

  std::string uuid = raw.substr(0, 8) + "-" + raw.substr(8, 4) + "-" + raw.substr(12, 4) + "-" +
                     raw.substr(16, 4) + "-" + raw.substr(20);
  return uuid;
}

/**
 * @brief Converts a uuid string to a valid gxf_tid
 *
 * @param uuid A valid uuid string of type 85f64c84-8236-4035-9b9a-3843a6a2026f
 * @return gxf_tid_t
 */
inline gxf_tid_t uuid_to_tid(std::string& uuid) {
  if (uuid.length() != 36) GXF_LOG_ERROR("Invalid uuid : %s ", uuid.c_str());

  size_t pos = 0;
  while (pos != std::string::npos) {
    pos = uuid.find("-");
    uuid = uuid.replace(pos, 1, "");
    pos = uuid.find("-");
  }

  if (uuid.length() != 32) GXF_LOG_ERROR("Invalid uuid : %s ", uuid.c_str());

  gxf_tid_t tid{hex_string_to_uint64(uuid.substr(0, 16)), hex_string_to_uint64(uuid.substr(16))};
  return tid;
}

/**
 * @brief Gives a new extension info object with memory allocated
 *        to fetch a max of @param count component tids
 *
 * @param count Expected number of components in the extension
 * @return gxf_extension_info_t
 */
gxf_extension_info_t new_extension_info(uint64_t count = 0) {
  count = count ? count : MAX_COMP_COUNT;
  gxf_extension_info_t info;
  info.components = new gxf_tid_t[count];
  info.num_components = count;
  return info;
}

/**
 * @brief Free the dynamic memory allocated in extension info
 *
 * @param info gxf_extension_info
 */
void destruct_extension_info(gxf_extension_info_t& info) {
  info.version = nullptr;
  info.num_components = 0;
  delete[] info.components;
}

/**
 * @brief Resize the dynamic memory allocated in extension info
 *
 * @param info gxf_extension_info to be resized
 * @param count new number of expected components
 */
void realloc_extension_info(gxf_extension_info_t& info, uint64_t count) {
  destruct_extension_info(info);
  info = new_extension_info(count);
}

/**
 * @brief Gives a new component info object with memory allocated
 *        to fetch a max of @param count parameter keys
 *
 * @param count Expected number of parameters in the component
 * @return gxf_component_info_t
 */
gxf_component_info_t new_component_info(uint64_t count = 0) {
  count = count ? count : MAX_PARAM_COUNT;
  gxf_component_info_t info;
  info.parameters = new const char*[count];
  info.num_parameters = count;
  return info;
}

/**
 * @brief Free the dynamic memory allocated in component info
 *
 * @param info gxf_component_info
 */
void destruct_component_info(gxf_component_info_t& info) {
  info.num_parameters = 0;
  delete[] info.parameters;
}

/**
 * @brief Resize the dyamic memory allocted in component info
 *
 * @param info gxf_component_info to be resized
 * @param count new number of expected parameters
 */
void realloc_component_info(gxf_component_info_t& info, uint64_t count) {
  destruct_component_info(info);
  info = new_component_info(count);
}

PYBIND11_MODULE(core_pybind, m) {
  (void)pybind11::detail::get_internals();
  Py_DECREF(PyImport_ImportModule("threading"));
  m.doc() = R"pbdoc(
        Python bridge for GXF CORE
        -----------------------

        .. currentmodule:: core

    )pbdoc";
  m.def("context_create", []() {
    gxf_context_t context;
    gxf_result_t result = GxfContextCreate(&context);
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
    return reinterpret_cast<uint64_t>(context);
  });
  m.def("context_destroy", [](uint64_t context) {
    gxf_result_t result = GxfContextDestroy(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });

  m.def(
      "load_extensions",
      [](uint64_t context, const std::vector<std::string>& extension_filenames,
         const std::vector<std::string>& manifest_filenames, const std::string& base_directory) {
        std::vector<const char*> extension_filenames_cstr(extension_filenames.size());
        for (size_t i = 0; i < extension_filenames.size(); i++) {
          extension_filenames_cstr[i] = extension_filenames[i].c_str();
        }

        std::vector<const char*> manifest_filenames_cstr(manifest_filenames.size());
        for (size_t i = 0; i < manifest_filenames.size(); i++) {
          manifest_filenames_cstr[i] = manifest_filenames[i].c_str();
        }

        const GxfLoadExtensionsInfo info{
            extension_filenames_cstr.empty() ? nullptr : extension_filenames_cstr.data(),
            static_cast<uint32_t>(extension_filenames_cstr.size()),
            manifest_filenames_cstr.empty() ? nullptr : manifest_filenames_cstr.data(),
            static_cast<uint32_t>(manifest_filenames_cstr.size()), base_directory.c_str()};

        const gxf_result_t result =
            GxfLoadExtensions(reinterpret_cast<gxf_context_t>(context), &info);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Loads GXF extension libaries", pybind11::arg("context"),
      pybind11::arg("extension_filenames") = std::vector<std::string>{},
      pybind11::arg("manifest_filenames") = std::vector<std::string>{},
      pybind11::arg("base_directory") = "");

  m.def(
      "get_extension_list",
      [](uint64_t context) {
        gxf_context_t ctx = reinterpret_cast<gxf_context_t>(context);
        gxf_runtime_info info;
        info.extensions = new gxf_tid_t[MAX_EXT_COUNT];
        info.num_extensions = MAX_EXT_COUNT;

        gxf_result_t code = GxfRuntimeInfo(ctx, &info);
        if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
          realloc_runtime_info(info, info.num_extensions);
          code = GxfRuntimeInfo(ctx, &info);
        }
        if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

        // Populate a py list of ext uuids
        auto result = pybind11::list();
        for (uint i = 0; i < info.num_extensions; ++i) {
          gxf_tid_t tid = info.extensions[i];
          std::string uuid = tid_to_uuid(tid);
          result.append(uuid);
        }
        destruct_runtime_info(info);
        return result;
      },
      "List loaded extensions", pybind11::arg("context"), pybind11::return_value_policy::reference);

  m.def(
      "get_extension_info",
      [](uint64_t context, std::string& uuid) {
        gxf_context_t ctx = reinterpret_cast<gxf_context_t>(context);
        gxf_tid_t tid = uuid_to_tid(uuid);
        gxf_extension_info_t info = new_extension_info();

        gxf_result_t code = GxfExtensionInfo(ctx, tid, &info);
        if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
          realloc_extension_info(info, info.num_components);
          code = GxfExtensionInfo(ctx, tid, &info);
        }
        if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

        auto result = pybind11::dict();
        result["name"] = info.name;
        result["display_name"] = info.display_name;
        result["category"] = info.category;
        result["brief"] = info.brief;
        result["description"] = info.description;
        result["version"] = info.version;
        result["author"] = info.author;
        result["license"] = info.license;
        destruct_extension_info(info);

        return result;
      },
      "Get extension info", pybind11::arg("context"), pybind11::arg("uuid"));

  m.def(
      "get_component_list",
      [](uint64_t context, std::string& uuid) {
        gxf_context_t ctx = reinterpret_cast<gxf_context_t>(context);
        gxf_tid_t tid = uuid_to_tid(uuid);
        gxf_extension_info_t info = new_extension_info();

        gxf_result_t code = GxfExtensionInfo(ctx, tid, &info);
        if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
          realloc_extension_info(info, info.num_components);
          code = GxfExtensionInfo(ctx, tid, &info);
        }
        if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

        // Populate a py list of component uuids
        auto result = pybind11::list();
        for (uint i = 0; i < info.num_components; ++i) {
          gxf_tid_t tid = info.components[i];
          std::string uuid = tid_to_uuid(tid);
          result.append(uuid);
        }
        destruct_extension_info(info);
        return result;
      },
      "Get list of components of an extension ", pybind11::arg("context"), pybind11::arg("uuid"),
      pybind11::return_value_policy::reference);

  m.def(
      "get_component_info",
      [](uint64_t context, std::string& uuid) {
        gxf_context_t ctx = reinterpret_cast<gxf_context_t>(context);
        gxf_tid_t tid = uuid_to_tid(uuid);
        gxf_component_info_t info = new_component_info();

        gxf_result_t code = GxfComponentInfo(ctx, tid, &info);
        if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
          realloc_component_info(info, info.num_parameters);
          code = GxfComponentInfo(ctx, tid, &info);
        }
        if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

        auto result = pybind11::dict();
        result["typename"] = info.type_name;
        result["display_name"] = info.display_name;
        result["brief"] = info.brief;
        result["description"] = info.description;
        result["base_typename"] = info.base_name == nullptr ? "" : info.base_name;
        result["is_abstract"] = info.is_abstract ? true : false;
        destruct_component_info(info);

        return result;
      },
      "Get component info", pybind11::arg("context"), pybind11::arg("uuid"),
      pybind11::return_value_policy::reference);

  m.def(
      "get_param_list",
      [](uint64_t context, std::string& uuid) {
        gxf_context_t ctx = reinterpret_cast<gxf_context_t>(context);
        gxf_tid_t tid = uuid_to_tid(uuid);
        gxf_component_info_t info = new_component_info();

        gxf_result_t code = GxfComponentInfo(ctx, tid, &info);
        if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
          realloc_component_info(info, info.num_parameters);
          code = GxfComponentInfo(ctx, tid, &info);
        }
        if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

        auto result = pybind11::list();
        for (uint i = 0; i < info.num_parameters; ++i) {
          std::string param_key(info.parameters[i]);
          result.append(param_key);
        }
        destruct_component_info(info);

        return result;
      },
      "Get parameter list of a component", pybind11::arg("context"), pybind11::arg("uuid"),
      pybind11::return_value_policy::reference);

  m.def(
      "get_param_info",
      [](uint64_t context, std::string& uuid, std::string& key) {
        gxf_context_t ctx = reinterpret_cast<gxf_context_t>(context);
        gxf_tid_t tid = uuid_to_tid(uuid);
        gxf_parameter_info_t info;

        gxf_result_t code = GxfParameterInfo(ctx, tid, key.c_str(), &info);
        if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

        auto result = pybind11::dict();
        result["key"] = info.key;
        result["headline"] = info.headline;
        result["description"] = info.description;
        result["gxf_parameter_type"] = GxfParameterTypeStr(info.type);
        result["rank"] = info.rank;
        auto shape = pybind11::list();
        if (info.rank > 0) {
          for (int i = 0; i < info.rank; ++i) { shape.append(info.shape[i]); }
        } else
          // shape 1 is for scalar when rank = 0
          shape.append(1);
        result["shape"] = shape;

        // FIXME: Flags can have multiple values
        if ((info.flags & GXF_PARAMETER_FLAGS_OPTIONAL) == GXF_PARAMETER_FLAGS_OPTIONAL)
          result["flags"] = GxfParameterFlagTypeStr(GXF_PARAMETER_FLAGS_OPTIONAL);
        else if ((info.flags & GXF_PARAMETER_FLAGS_DYNAMIC) == GXF_PARAMETER_FLAGS_DYNAMIC)
          result["flags"] = GxfParameterFlagTypeStr(GXF_PARAMETER_FLAGS_DYNAMIC);
        else
          result["flags"] = GxfParameterFlagTypeStr(GXF_PARAMETER_FLAGS_NONE);

        if (!GxfTidIsNull(info.handle_tid)) {
          gxf_tid_t tid = info.handle_tid;
          const char* type_name = nullptr;
          code = GxfComponentTypeName(ctx, tid, &type_name);
          if (code != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(code)); }

          result["handle_type"] = type_name;
        } else
          result["handle_type"] = "N/A";

        // Default value
        if (info.default_value) {
          if (info.type == GXF_PARAMETER_TYPE_INT8) {
            int8_t value = *static_cast<const int8_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
            int16_t value = *static_cast<const int16_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT32) {
            int32_t value = *static_cast<const int32_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT64) {
            int64_t value = *static_cast<const int64_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT8) {
            uint8_t value = *static_cast<const uint8_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
            uint16_t value = *static_cast<const uint16_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT32) {
            uint32_t value = *static_cast<const uint32_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
            uint64_t value = *static_cast<const uint64_t*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT32) {
            float value = *static_cast<const float*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT64) {
            double value = *static_cast<const double*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_STRING) {
            result["default"] = std::string(static_cast<const char*>(info.default_value));
          } else if (info.type == GXF_PARAMETER_TYPE_BOOL) {
            bool value = *static_cast<const bool*>(info.default_value);
            result["default"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FILE) {
            result["default"] = std::string(static_cast<const char*>(info.default_value));
          } else {
            result["default"] = "N/A";
          }
        } else {
          result["default"] = "N/A";
        }

        // Max value
        if (info.numeric_max) {
          if (info.type == GXF_PARAMETER_TYPE_UINT8) {
            uint8_t value = *static_cast<const uint8_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
            uint16_t value = *static_cast<const uint16_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT32) {
            uint32_t value = *static_cast<const uint32_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
            uint64_t value = *static_cast<const uint64_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT8) {
            int8_t value = *static_cast<const int8_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
            int16_t value = *static_cast<const int16_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT32) {
            int32_t value = *static_cast<const int32_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT64) {
            int64_t value = *static_cast<const int64_t*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT32) {
            float value = *static_cast<const float*>(info.numeric_max);
            result["max_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT64) {
            double value = *static_cast<const double*>(info.numeric_max);
            result["max_value"] = value;
          }
        }

        // Min value
        if (info.numeric_min) {
          if (info.type == GXF_PARAMETER_TYPE_UINT8) {
            uint8_t value = *static_cast<const uint8_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
            uint16_t value = *static_cast<const uint16_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT32) {
            uint32_t value = *static_cast<const uint32_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
            uint64_t value = *static_cast<const uint64_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT8) {
            int8_t value = *static_cast<const int8_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
            int16_t value = *static_cast<const int16_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT32) {
            int32_t value = *static_cast<const int32_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT64) {
            int64_t value = *static_cast<const int64_t*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT32) {
            float value = *static_cast<const float*>(info.numeric_min);
            result["min_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT64) {
            double value = *static_cast<const double*>(info.numeric_min);
            result["min_value"] = value;
          }
        }

        // Step value
        if (info.numeric_step) {
          if (info.type == GXF_PARAMETER_TYPE_UINT8) {
            uint8_t value = *static_cast<const uint8_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
            uint16_t value = *static_cast<const uint16_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT32) {
            uint32_t value = *static_cast<const uint32_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
            uint64_t value = *static_cast<const uint64_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT8) {
            int8_t value = *static_cast<const int8_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
            int16_t value = *static_cast<const int16_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT32) {
            int32_t value = *static_cast<const int32_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_INT64) {
            int64_t value = *static_cast<const int64_t*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT32) {
            float value = *static_cast<const float*>(info.numeric_step);
            result["step_value"] = value;
          } else if (info.type == GXF_PARAMETER_TYPE_FLOAT64) {
            double value = *static_cast<const double*>(info.numeric_step);
            result["step_value"] = value;
          }
        }

        return result;
      },
      "Get parameter info", pybind11::arg("context"), pybind11::arg("uuid"), pybind11::arg("param"),
      pybind11::return_value_policy::reference);

  m.def("graph_load_file", [](uint64_t context, const char* filename) {
    gxf_result_t result = GxfGraphLoadFile(reinterpret_cast<gxf_context_t>(context), filename);
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("set_root_filepath", [](uint64_t context, const char* filename) {
    gxf_result_t result = GxfGraphSetRootPath(reinterpret_cast<gxf_context_t>(context), filename);
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("_subgraph_load_file", [](uint64_t context, const char* filename, const char* entity_prefix, gxf_uid_t parent_eid, const char* prerequisites) {
    YAML::Node yaml_node = YAML::Load(std::string(prerequisites));
    gxf_result_t result = GxfGraphLoadFileExtended(reinterpret_cast<gxf_context_t>(context), filename, entity_prefix, nullptr, 0, parent_eid, static_cast<void*>(&yaml_node));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_activate", [](uint64_t context) {
    gxf_result_t result = GxfGraphActivate(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_run", [](uint64_t context) {
    pybind11::gil_scoped_release release_gil;
    gxf_result_t result = GxfGraphRun(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_run_async", [](uint64_t context) {
    gxf_result_t result = GxfGraphRunAsync(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_interrupt", [](uint64_t context) {
    pybind11::gil_scoped_release release_gil;
    gxf_result_t result = GxfGraphInterrupt(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_wait", [](uint64_t context) {
    pybind11::gil_scoped_release release_gil;
    gxf_result_t result = GxfGraphWait(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_save", [](uint64_t context, const char* filename) {
    gxf_result_t result = GxfGraphSaveToFile(reinterpret_cast<gxf_context_t>(context), filename);
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def("graph_deactivate", [](uint64_t context) {
    gxf_result_t result = GxfGraphDeactivate(reinterpret_cast<gxf_context_t>(context));
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
  });
  m.def(
      "gxf_set_severity",
      [](uint64_t context, uint32_t severity) {
        gxf_result_t result = GxfSetSeverity(reinterpret_cast<gxf_context_t>(context),
                                             static_cast<gxf_severity_t>(severity));
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Severity levels for GXF_LOG_* logging macros", pybind11::arg("context"),
      pybind11::arg("severity"));
  m.def("entity_find", [](uint64_t context, const char* entity_name) {
    gxf_uid_t eid = kNullUid;
    const gxf_result_t result =
        GxfEntityFind(reinterpret_cast<gxf_context_t>(context), entity_name, &eid);
    if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
    return eid;
  });
  m.def("get_gxf_primitive_type", [](pybind11::dtype type) {
    if (type.is(pybind11::dtype::of<std::int8_t>())) {
      return "nvidia::gxf::PrimitiveType::kInt8";
    } else if (type.is(pybind11::dtype::of<std::int16_t>())) {
      return "nvidia::gxf::PrimitiveType::kInt16";
    } else if (type.is(pybind11::dtype::of<std::int32_t>())) {
      return "nvidia::gxf::PrimitiveType::kInt32";
    } else if (type.is(pybind11::dtype::of<std::int64_t>())) {
      return "nvidia::gxf::PrimitiveType::kInt64";
    } else if (type.is(pybind11::dtype::of<std::uint8_t>())) {
      return "nvidia::gxf::PrimitiveType::kUInt8";
    } else if (type.is(pybind11::dtype::of<std::uint16_t>())) {
      return "nvidia::gxf::PrimitiveType::kUInt16";
    } else if (type.is(pybind11::dtype::of<std::uint32_t>())) {
      return "nvidia::gxf::PrimitiveType::kUInt32";
    } else if (type.is(pybind11::dtype::of<std::uint64_t>())) {
      return "nvidia::gxf::PrimitiveType::kUInt64";
    } else if (type.is(pybind11::dtype::of<std::float_t>())) {
      return "nvidia::gxf::PrimitiveType::kFloat32";
    } else if (type.is(pybind11::dtype::of<std::double_t>())) {
      return "nvidia::gxf::PrimitiveType::kFloat64";
    }
    return "nvidia::gxf::PrimitiveType::kCustom";
  });
  m.def(
      "entity_find_all",
      [](uint64_t context) {
        uint64_t num_entities = 1024;
        gxf_uid_t* entities = new gxf_uid_t[1024];
        const gxf_result_t result =
            GxfEntityFindAll(reinterpret_cast<gxf_context_t>(context), &num_entities, entities);
        if (result != GXF_SUCCESS) {
          std::cout << "result" << GxfResultStr(result) << "\n";
          throw pybind11::value_error(GxfResultStr(result));
        }
        std::cout << "num_entities" << num_entities << "\n";
        pybind11::list output;
        for (uint i = 0; i < num_entities; i++) { output.append(entities + i); }
        return output;
      },
      "Get all the entities of the context", pybind11::arg("context"),
      pybind11::return_value_policy::reference);
  m.def(
      "parameter_set_1d_float64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<double> value,
         uint64_t length) {
        if (value.size() != length)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        const gxf_result_t result = GxfParameterSet1DFloat64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value.data(), length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 64-bit float (double) 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("length"));
  m.def(
      "parameter_set_2d_float64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<std::vector<double>> value,
         uint64_t height, uint64_t width) {
        double* value_[height];
        if (value.size() != height)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        for (uint i = 0; i < height; i++) {
          if (value[i].size() != width)
            throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
          value_[i] = value[i].data();
        }
        const gxf_result_t result = GxfParameterSet2DFloat64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, height, width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 64-bit float (double) 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("height"),
      pybind11::arg("width"));
  m.def(
      "parameter_set_1d_int64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<int64_t> value,
         uint64_t length) {
        if (value.size() != length)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        const gxf_result_t result = GxfParameterSet1DInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value.data(), length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 64-bit int 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("length"));
  m.def(
      "parameter_set_2d_int64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<std::vector<int64_t>> value,
         uint64_t height, uint64_t width) {
        int64_t* value_[height];
        if (value.size() != height)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        for (uint i = 0; i < height; i++) {
          if (value[i].size() != width)
            throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
          value_[i] = value[i].data();
        }
        const gxf_result_t result = GxfParameterSet2DInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, height, width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 64-bit signed int 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("height"),
      pybind11::arg("width"));
  m.def(
      "parameter_set_1d_uint64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<uint64_t> value,
         uint64_t length) {
        if (value.size() != length)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        const gxf_result_t result = GxfParameterSet1DUInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value.data(), length);

        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a unsigned 64-bit int 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("length"));
  m.def(
      "parameter_set_2d_uint64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<std::vector<uint64_t>> value,
         uint64_t height, uint64_t width) {
        uint64_t* value_[height];
        if (value.size() != height)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        for (uint i = 0; i < height; i++) {
          if (value[i].size() != width)
            throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
          value_[i] = value[i].data();
        }
        const gxf_result_t result = GxfParameterSet2DUInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, height, width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 64-bit unsigned int 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("height"),
      pybind11::arg("width"));
  m.def(
      "parameter_set_1d_int32_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<int32_t> value,
         uint64_t length) {
        if (value.size() != length)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        const gxf_result_t result = GxfParameterSet1DInt32Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value.data(), length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 32-bit signed int 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("length"));
  m.def(
      "parameter_set_2d_int32_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, std::vector<std::vector<int32_t>> value,
         uint64_t height, uint64_t width) {
        int32_t* value_[height];
        if (value.size() != height)
          throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
        for (uint i = 0; i < height; i++) {
          if (value[i].size() != width)
            throw pybind11::value_error(GxfResultStr(GXF_ARGUMENT_OUT_OF_RANGE));
          value_[i] = value[i].data();
        }
        const gxf_result_t result = GxfParameterSet2DInt32Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, height, width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a 32-bit signed int 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("height"),
      pybind11::arg("width"));

  m.def(
      "parameter_get_1d_float64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t length) {
        double value[length];
        const gxf_result_t result = GxfParameterGet1DFloat64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value, &length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < length; i++) output.append(value[i]);
        return output;
      },
      "Get a 64-bit floating point 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("length"));
  m.def(
      "parameter_get_2d_float64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t height, uint64_t width) {
        double value_inner[height * width];
        double* value_[height];
        for (uint i = 0; i < height; i++) { value_[i] = value_inner + i * width; }
        const gxf_result_t result = GxfParameterGet2DFloat64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, &height, &width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < height; i++) {
          pybind11::list value_i;
          for (uint j = 0; j < width; j++) { value_i.append(value_[i][j]); }
          output.append(value_i);
        }
        return output;
      },
      "Get a 64-bit floating point 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("height"), pybind11::arg("width"));
  m.def(
      "parameter_get_1d_int64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t length) {
        int64_t value[length];
        const gxf_result_t result = GxfParameterGet1DInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value, &length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < length; i++) output.append(value[i]);
        return output;
      },
      "Get a 64-bit int 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("length"));
  m.def(
      "parameter_get_2d_int64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t height, uint64_t width) {
        int64_t value_inner[height * width];
        int64_t* value_[height];
        for (uint i = 0; i < height; i++) { value_[i] = value_inner + i * width; }
        const gxf_result_t result = GxfParameterGet2DInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, &height, &width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < height; i++) {
          pybind11::list value_i;
          for (uint j = 0; j < width; j++) { value_i.append(value_[i][j]); }
          output.append(value_i);
        }
        return output;
      },
      "Get a 64-bit signed integer 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("height"), pybind11::arg("width"));
  m.def(
      "parameter_get_1d_uint64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t length) {
        uint64_t value[length];
        const gxf_result_t result = GxfParameterGet1DUInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value, &length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < length; i++) output.append(value[i]);
        return output;
      },
      "Get a unsigned 64-bit int 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("length"));
  m.def(
      "parameter_get_2d_uint64_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t height, uint64_t width) {
        uint64_t value_inner[height * width];
        uint64_t* value_[height];
        for (uint i = 0; i < height; i++) { value_[i] = value_inner + i * width; }
        const gxf_result_t result = GxfParameterGet2DUInt64Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, &height, &width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < height; i++) {
          pybind11::list value_i;
          for (uint j = 0; j < width; j++) { value_i.append(value_[i][j]); }
          output.append(value_i);
        }
        return output;
      },
      "Get a 64-bit unsigned integer 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("height"), pybind11::arg("width"));
  m.def(
      "parameter_get_1d_int32_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t length) {
        int32_t value[length];
        const gxf_result_t result = GxfParameterGet1DInt32Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value, &length);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < length; i++) output.append(value[i]);
        return output;
      },
      "Get a signed 32-bit int 1-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("length"));
  m.def(
      "parameter_get_2d_int32_vector",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t height, uint64_t width) {
        int32_t value_inner[height * width];
        int32_t* value_[height];
        for (uint i = 0; i < height; i++) { value_[i] = value_inner + i * width; }
        const gxf_result_t result = GxfParameterGet2DInt32Vector(
            reinterpret_cast<gxf_context_t>(context), eid, key, value_, &height, &width);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        pybind11::list output;
        for (uint i = 0; i < height; i++) {
          pybind11::list value_i;
          for (uint j = 0; j < width; j++) { value_i.append(value_[i][j]); }
          output.append(value_i);
        }
        return output;
      },
      "Get a 32-bit signed integer 2-dimensional vector parameter", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("key"), pybind11::arg("height"), pybind11::arg("width"));
  m.def(
      "entity_create",
      [](uint64_t context, GxfEntityCreateInfo* info) {
        gxf_uid_t eid;
        const gxf_result_t result =
            GxfCreateEntity(reinterpret_cast<gxf_context_t>(context), info, &eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return eid;
      },
      "Create an entity with info", pybind11::arg("context"), pybind11::arg("info"));
  m.def(
      "entity_group_create",
      [](uint64_t context, const std::string& name) {
        gxf_uid_t gid;
        const gxf_result_t result =
            GxfCreateEntityGroup(reinterpret_cast<gxf_context_t>(context), name.c_str(), &gid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return gid;
      },
      "Create an entity group with name", pybind11::arg("context"), pybind11::arg("name") = "");
  m.def(
      "entity_group_add",
      [](uint64_t context, gxf_uid_t gid, gxf_uid_t eid) {
        const gxf_result_t result =
            GxfUpdateEntityGroup(reinterpret_cast<gxf_context_t>(context), gid, eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return gid;
      },
      "Add an entity to an entity group", pybind11::arg("context"), pybind11::arg("gid"), pybind11::arg("eid"));
  m.def(
      "entity_activate",
      [](uint64_t context, gxf_uid_t eid) {
        const gxf_result_t result =
            GxfEntityActivate(reinterpret_cast<gxf_context_t>(context), eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Activate an entity", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "entity_deactivate",
      [](uint64_t context, gxf_uid_t eid) {
        const gxf_result_t result =
            GxfEntityDeactivate(reinterpret_cast<gxf_context_t>(context), eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Deactivate entity", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "entity_destroy",
      [](uint64_t context, gxf_uid_t eid) {
        const gxf_result_t result = GxfEntityDestroy(reinterpret_cast<gxf_context_t>(context), eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Destroy an entity", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "entity_ref_count_inc",
      [](uint64_t context, gxf_uid_t eid) {
        const gxf_result_t result =
            GxfEntityRefCountInc(reinterpret_cast<gxf_context_t>(context), eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Increase reference count of an entity", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "entity_ref_count_dec",
      [](uint64_t context, gxf_uid_t eid) {
        const gxf_result_t result =
            GxfEntityRefCountDec(reinterpret_cast<gxf_context_t>(context), eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Decrease reference count of an entity", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "entity_get_state",
      [](uint64_t context, gxf_uid_t eid) {
        entity_state_t state;
        const gxf_result_t result =
            GxfEntityGetState(reinterpret_cast<gxf_context_t>(context), eid, &state);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return state;
      },
      "Get state of an entity", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "entity_event_notify",
      [](uint64_t context, gxf_uid_t eid) {
        const gxf_result_t result =
            GxfEntityEventNotify(reinterpret_cast<gxf_context_t>(context), eid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Notify entity on an event", pybind11::arg("context"), pybind11::arg("eid"));
  m.def(
      "component_add",
      [](uint64_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name) {
        gxf_uid_t cid = kNullUid;
        const gxf_result_t result =
            GxfComponentAdd(reinterpret_cast<gxf_context_t>(context), eid, tid, name, &cid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return cid;
      },
      "Add component of type_id `tid` to  an entity with id `eid`", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("tid"), pybind11::arg("name"));

  pybind11::class_<GxfEntityCreateInfo>(m, "gxf_entity_create_info")
      .def(pybind11::init([](const char* name, uint32_t flags) {
             // doesn't work without copy of the char * variable
             char name_copy[2048];
             if((name == NULL) || (name[0] == '\0')){
              return new GxfEntityCreateInfo{NULL, flags};
             }
             else {
              strcpy(name_copy, name);
              return new GxfEntityCreateInfo{name_copy, flags};
             }
           }),
           "Constructor for gxf.core.gxf_entity_create_info", pybind11::arg("name"),
           pybind11::arg("flags"), pybind11::return_value_policy::reference)
      .def_readonly("entity_name", &GxfEntityCreateInfo::entity_name)
      .def_readonly("flags", &GxfEntityCreateInfo::flags);

  pybind11::class_<gxf_uid_t>(m, "gxf_uid_t").def(pybind11::init<int64_t>());

  pybind11::class_<gxf_tid_t>(m, "gxf_tid_t")
      .def(pybind11::init<>())
      .def_readwrite("hash1", &gxf_tid_t::hash1)
      .def_readwrite("hash2", &gxf_tid_t::hash2);
  m.def("tid_null", []() { return GxfTidNull(); });
  m.def(
      "component_find",
      [](uint64_t context, gxf_uid_t eid, gxf_tid_t tid, const char* component_name) {
        std::vector<gxf_uid_t> cids;
        for (int offset = 0;; offset++) {
          gxf_uid_t cid = kNullUid;
          const gxf_result_t result = GxfComponentFind(reinterpret_cast<gxf_context_t>(context),
                                                       eid, tid, component_name, &offset, &cid);
          if (result == GXF_ENTITY_COMPONENT_NOT_FOUND) {
            // We found all components
            return cids;
          } else if (result != GXF_SUCCESS) {
            throw pybind11::value_error(GxfResultStr(result));
          }
          cids.emplace_back(cid);
          // Continue the loop to find other components
        }
      },
      "Finds a component in an entity", pybind11::arg("context"), pybind11::arg("eid"),
      pybind11::arg("tid") = GxfTidNull(), pybind11::arg("component_name"));

  m.def(
      "component_type_id",
      [](uint64_t context, const char* component_name) {
        gxf_tid_t tid;
        const gxf_result_t result =
            GxfComponentTypeId(reinterpret_cast<gxf_context_t>(context), component_name, &tid);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return tid;
      },
      "Get component type id. Return value is of type gxf.core.gxf_tid_t", pybind11::arg("context"),
      pybind11::arg("component_name"));

  m.def(
      "component_type_name",
      [](uint64_t context, gxf_tid_t tid) {
        const char* type_name;
        const gxf_result_t result =
            GxfComponentTypeName(reinterpret_cast<gxf_context_t>(context), tid, &type_name);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return std::string(type_name);
      },
      "Get component type name. Return name of type corresponding to tid", pybind11::arg("context"),
      pybind11::arg("tid"));

  m.def(
      "component_add_to_interface",
      [](uint64_t context, gxf_uid_t eid, gxf_uid_t cid, const char* name) {
        const gxf_result_t result =
            GxfComponentAddToInterface(reinterpret_cast<gxf_context_t>(context), eid, cid, name);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return;
      },
      "Adds the component to the alias map", pybind11::arg("context"),
      pybind11::arg("eid"), pybind11::arg("cid"), pybind11::arg("name"));

  m.def(
      "parameter_set_float64",
      [](uint64_t context, gxf_uid_t eid, const char* key, double value) {
        const gxf_result_t result =
            GxfParameterSetFloat64(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a float64 (double) parameter", pybind11::arg("context"),
      pybind11::arg("eid") = GxfTidNull(), pybind11::arg("key"), pybind11::arg("value"));
  m.def(
      "parameter_set_float32",
      [](uint64_t context, gxf_uid_t eid, const char* key, float value) {
        const gxf_result_t result =
            GxfParameterSetFloat32(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a float parameter", pybind11::arg("context"),
      pybind11::arg("eid") = GxfTidNull(), pybind11::arg("key"), pybind11::arg("value"));
  m.def(
      "parameter_set_int64",
      [](uint64_t context, gxf_uid_t eid, const char* key, int64_t value) {
        const gxf_result_t result =
            GxfParameterSetInt64(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set an int64 parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_uint64",
      [](uint64_t context, gxf_uid_t eid, const char* key, uint64_t value) {
        const gxf_result_t result =
            GxfParameterSetUInt64(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a uint64 parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_str",
      [](uint64_t context, gxf_uid_t eid, const char* key, const char* value) {
        const gxf_result_t result =
            GxfParameterSetStr(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a string parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_handle",
      [](uint64_t context, gxf_uid_t eid, const char* key, gxf_uid_t value) {
        const gxf_result_t result =
            GxfParameterSetHandle(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a handle parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_bool",
      [](uint64_t context, gxf_uid_t eid, const char* key, bool value) {
        const gxf_result_t result =
            GxfParameterSetBool(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a boolean parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_path",
      [](uint64_t context, gxf_uid_t eid, const char* key, const char* value) {
        const gxf_result_t result =
            GxfParameterSetPath(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set a string parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_int32",
      [](uint64_t context, gxf_uid_t eid, const char* key, int32_t value) {
        const gxf_result_t result =
            GxfParameterSetInt32(reinterpret_cast<gxf_context_t>(context), eid, key, value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set an int32 parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"), pybind11::arg("value"));

  m.def(
      "parameter_set_from_yaml_node",
      [](uint64_t context, gxf_uid_t eid, const char* key, const char* input) {
        YAML::Node yaml_node = YAML::Load(std::string(input));
        const gxf_result_t result = GxfParameterSetFromYamlNode(
            reinterpret_cast<gxf_context_t>(context), eid, key, static_cast<void*>(&yaml_node), "");
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
      },
      "Set parameters using a yaml string", pybind11::arg("context"),
      pybind11::arg("eid") = GxfTidNull(), pybind11::arg("key"), pybind11::arg("input"));

  m.def(
      "parameter_get_float64",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        double value;
        const gxf_result_t result =
            GxfParameterGetFloat64(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get a float64 (double) parameter", pybind11::arg("context"),
      pybind11::arg("eid") = GxfTidNull(), pybind11::arg("key"));

  m.def(
      "parameter_get_float32",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        float value;
        const gxf_result_t result =
            GxfParameterGetFloat32(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get a float32 parameter", pybind11::arg("context"),
      pybind11::arg("eid") = GxfTidNull(), pybind11::arg("key"));

  m.def(
      "parameter_get_int64",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        int64_t value;
        const gxf_result_t result =
            GxfParameterGetInt64(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get an int64 parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"));

  m.def(
      "parameter_get_uint64",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        uint64_t value;
        const gxf_result_t result =
            GxfParameterGetUInt64(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get a uint64 parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"));

  m.def(
      "parameter_get_str",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        const char* value;
        const gxf_result_t result =
            GxfParameterGetStr(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return std::string(value);
      },
      "Get a string parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"));

  m.def(
      "parameter_get_handle",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        gxf_uid_t value;
        const gxf_result_t result =
            GxfParameterGetHandle(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get a handle parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"));

  m.def(
      "parameter_get_bool",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        bool value;
        const gxf_result_t result =
            GxfParameterGetBool(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get a boolean parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"));

  m.def(
      "parameter_get_int32",
      [](uint64_t context, gxf_uid_t eid, const char* key) {
        int32_t value;
        const gxf_result_t result =
            GxfParameterGetInt32(reinterpret_cast<gxf_context_t>(context), eid, key, &value);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return value;
      },
      "Get an int32 parameter", pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull(),
      pybind11::arg("key"));

  m.def(
      "entity_get_status",
      [](uint64_t context, gxf_uid_t eid) {
        gxf_entity_status_t entity_status;
        const gxf_result_t result =
            GxfEntityGetStatus(reinterpret_cast<gxf_context_t>(context), eid, &entity_status);
        if (result != GXF_SUCCESS) { throw pybind11::value_error(GxfResultStr(result)); }
        return static_cast<int64_t>(entity_status);
      },
      "Gets the status of a entity. 0=ENTITY_NOT_STARTED\n, 1=ENTITY_START_PENDING\n,\
       2=ENTITY_STARTED\n, 3=ENTITY_TICK_PENDING\n, 4=ENTITY_STOP_PENDING\n",
      pybind11::arg("context"), pybind11::arg("eid") = GxfTidNull());

  pybind11::class_<nvidia::gxf::Entity>(m, "MessageEntity")
      .def(pybind11::init([](gxf_context_t context) {
        auto entity = nvidia::gxf::Entity::New(context);
        return entity.value();
      }))
      .def("eid", [](nvidia::gxf::Entity& entity) { return entity.eid(); });
}
