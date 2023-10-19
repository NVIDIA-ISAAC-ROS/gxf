/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef PYGXF_HPP
#define PYGXF_HPP

#include <sstream>

#include "common/logger.hpp"
#include "gxf/core/gxf.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#define MAX_EXT_COUNT 100
#define MAX_COMP_COUNT 300
#define MAX_PARAM_COUNT 500

#define GXF_RESULT_CHECK(code) \
  if (code != GXF_SUCCESS) GXF_LOG_ERROR("Error: %s", GxfResultStr(code));

namespace py = pybind11;

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
 * @brief Converts a gxf_tid to a uuid string
 *
 * @param tid A valid tid from gxf runtime
 * @return std::string UUID of type 85f64c84-8236-4035-9b9a-3843a6a2026f
 */
inline std::string tid_to_uuid(gxf_tid_t& tid) {
  std::string raw(uint64_to_hex_string(tid.hash1) +
                  uint64_to_hex_string(tid.hash2));
  if (raw.length() != 32) GXF_LOG_ERROR("Invalid tid : %s ", raw.c_str());

  std::string uuid = raw.substr(0, 8) + "-" + raw.substr(8, 4) + "-" +
                     raw.substr(12, 4) + "-" + raw.substr(16, 4) + "-" +
                     raw.substr(20);
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

  gxf_tid_t tid{hex_string_to_uint64(uuid.substr(0, 16)),
                hex_string_to_uint64(uuid.substr(16))};
  return tid;
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

/**
 * @brief Create a string corresponding to the result state
 *
 * @param result
 * @return std::string
 */
std::string gxf_result_str(gxf_result_t result) {
  return std::string(GxfResultStr(result));
}

/**
 * @brief Creates a gxf context and returns a capsule object with
 *        the allocated runtime context. User is responsible to
 *        destroy the context using gxf_context_destroy()
 *
 * @return py::capsule gxf context allocated by gxf core
 */
py::object gxf_context_create() {
  gxf_context_t ctx;
  gxf_result_t code = GxfContextCreate(&ctx);
  if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }
  return py::capsule(ctx, "gxf context in capsule");
}

/**
 * @brief Destorys a context which was created using gxf_context_create()
 *
 * @param context py::capsule object containting a pointer to a valid gxf
 * context
 */
bool gxf_context_destroy(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_result_t result = GxfContextDestroy(ctx);
  return result == GXF_SUCCESS ? true : false;
}

/**
 * @brief Registers a component type which can be used for GXF core
 *
 * @param context gxf context created using gxf_context_create()
 * @param uuid uuid sting of format 85f64c84-8236-4035-9b9a-3843a6a2026f
 * @param name name of the component
 * @param base_name valid name of the base type component which has been
 registered with the context
 */
void gxf_register_component(py::capsule& context, std::string& uuid,
                            std::string& name, std::string& base_name) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_tid_t tid = uuid_to_tid(uuid);
  GXF_RESULT_CHECK(
      GxfRegisterComponent(ctx, tid, name.c_str(), base_name.c_str()));
}

/**
 * @brief Loads an extension library
 *
 * @param context gxf context created using gxf_context_create()
 * @param filename path to an extension library
 */
bool gxf_load_ext(py::capsule& context, std::string& filename) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_result_t result = GxfLoadExtension(ctx, filename.c_str());
  return result == GXF_SUCCESS ? true : false;
}

/**
 * @brief Loads multiple extensions as specified in the manifest file. The
 * manifest file is a YAML file which lists extensions to be loaded.
 *
 * @param context gxf context created using gxf_context_create()
 * @param filename path to a manifest yaml file
 *                 Sample Manifest file format -
 *                 extensions:
 *                 - /path/to/extension_1.so
 *                 - /path/to/extension_2.so
 */
bool gxf_load_ext_manifest(py::capsule& context, std::string& filename) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_result_t result = GxfLoadExtensionManifest(ctx, filename.c_str());
  return result == GXF_SUCCESS ? true : false;
}

/**
 * @brief Loads a list of extensions using the "GxfLoadExtensions" api in gxf core
 *
 * @param context gxf context created using gxf_context_create()
 * @param filenames list of extension shared library filename
 */
bool gxf_load_extensions(py::capsule& context, std::vector<std::string>& filenames) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  std::vector<const char*> ext_filnames(filenames.size());
  for(size_t i = 0; i < filenames.size(); ++i) {
    ext_filnames[i] = filenames[i].c_str();
  }

  GxfLoadExtensionsInfo ext_info{
    ext_filnames.data(), static_cast<uint32_t>(filenames.size()),
    nullptr, 0, nullptr
  };
  gxf_result_t result = GxfLoadExtensions(ctx, &ext_info);
  return result == GXF_SUCCESS ? true : false;
}

/**
 * @brief Loads a list of extension metadata using the "GxfLoadExtensionMetadataFiles"
 *        api in gxf core
 *
 * @param context gxf context created using gxf_context_create()
 * @param filenames list of extension shared library filename
 */
bool gxf_load_extension_metadata(py::capsule& context, std::vector<std::string>& filenames) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  std::vector<const char*> ext_filnames(filenames.size());
  for(size_t i = 0; i < filenames.size(); ++i) {
    ext_filnames[i] = filenames[i].c_str();
  }

  gxf_result_t result = GxfLoadExtensionMetadataFiles(ctx, ext_filnames.data(),
                                                      ext_filnames.size());
  return result == GXF_SUCCESS ? true : false;
}

/**
 * @brief Loads a list of entities from a YAML file.
 *
 * @param context gxf context created using gxf_context_create()
 * @param filename path to application yaml which has a list of entities
 */
void gxf_load_graph_file(py::capsule& context, std::string& filename) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphLoadFile(ctx, filename.c_str()));
}

/**
 * @brief Runs all System components and waits for their completion
 *
 * @param context gxf context created using gxf_context_create()
 */
void gxf_graph_run(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphRun(ctx));
}

/**
 * @brief Activate all System components
 *
 * @param context gxf context created using gxf_context_create()
 */
void gxf_graph_activate(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphActivate(ctx));
}

/**
 * @brief Deactivate all System components
 *
 * @param context gxf context created using gxf_context_create()
 */
void gxf_graph_deactivate(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphDeactivate(ctx));
}

/**
 * @brief Starts the execution of the graph asynchronously
 *
 * @param context gxf context created using gxf_context_create()
 */
void gxf_graph_run_async(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphRunAsync(ctx));
}

/**
 * @brief Interrupt the execution of the graph
 *
 * @param context gxf context created using gxf_context_create()
 */
void gxf_graph_interrupt(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphInterrupt(ctx));
}

/**
 * @brief Waits for the graph to complete execution
 *
 * @param context gxf context created using gxf_context_create()
 */
void gxf_graph_wait(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  GXF_RESULT_CHECK(GxfGraphWait(ctx));
}

/**
 * @brief Get the gxf core runtime version
 *
 * @param context gxf context created using gxf_context_create()
 * @return std::string version of gxf core runtime
 */
std::string get_runtime_version(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_runtime_info info = new_runtime_info();

  gxf_result_t code = GxfRuntimeInfo(ctx, &info);
  if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
    realloc_runtime_info(info, info.num_extensions);
    code = GxfRuntimeInfo(ctx, &info);
    GXF_RESULT_CHECK(code);
  }
  GXF_RESULT_CHECK(code);

  std::string version{info.version};
  destruct_runtime_info(info);
  return version;
}

/**
 * @brief Get the list of extensions loaded in the gxf context
 *
 * @param context gxf context created using gxf_context_create()
 * @return py::list list of extension uuids
 *         Sample UUID format - 85f64c84-8236-4035-9b9a-3843a6a2026f
 */
py::object get_ext_list(py::capsule& context) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_runtime_info info = new_runtime_info();

  gxf_result_t code = GxfRuntimeInfo(ctx, &info);
  if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
    realloc_runtime_info(info, info.num_extensions);
    code = GxfRuntimeInfo(ctx, &info);
  }
  if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }

  // Populate a py list of ext uuids
  auto result = py::list();
  for (uint i = 0; i < info.num_extensions; ++i) {
    gxf_tid_t tid = info.extensions[i];
    std::string uuid = tid_to_uuid(tid);
    result.append(uuid);
  }

  destruct_runtime_info(info);
  return result;
}

/**
 * @brief Get extension specific metadata such as name, description, version,
 * author and  license
 *
 * @param context gxf context created using gxf_context_create()
 * @param uuid valid extension uuid obtained using get_ext_list()
 * @return py::dict Dict containing extension metatdata
 */
py::object get_ext_info(py::capsule& context, std::string& uuid) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_tid_t tid = uuid_to_tid(uuid);
  gxf_extension_info_t info = new_extension_info();

  gxf_result_t code = GxfExtensionInfo(ctx, tid, &info);
  if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
    realloc_extension_info(info, info.num_components);
    code = GxfExtensionInfo(ctx, tid, &info);
  }
  if (code != GXF_SUCCESS) { return pybind11::cast<pybind11::none>(Py_None); }

  auto result = py::dict();
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
}

/**
 * @brief Get the list of components loaded in extension specified using uuid
 *
 * @param context gxf context created using gxf_context_create()
 * @param uuid valid extension uuid obtained using get_ext_list()
 * @return py::list list of component uuids
 *         Sample UUID format - 85f64c84-8236-4035-9b9a-3843a6a2026f
 */
py::object get_comp_list(py::capsule& context, std::string& uuid) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_tid_t tid = uuid_to_tid(uuid);
  gxf_extension_info_t info = new_extension_info();

  gxf_result_t code = GxfExtensionInfo(ctx, tid, &info);
  if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
    realloc_extension_info(info, info.num_components);
    code = GxfExtensionInfo(ctx, tid, &info);
  }
  if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }

  // Populate a py list of component uuids
  auto result = py::list();
  for (uint i = 0; i < info.num_components; ++i) {
    gxf_tid_t tid = info.components[i];
    std::string uuid = tid_to_uuid(tid);
    result.append(uuid);
  }
  destruct_extension_info(info);

  return result;
}

/**
 * @brief Get component specific metadata such as typename, description,
 * base_typename and is_abstract
 *
 * @param context gxf context created using gxf_context_create()
 * @param uuid valid component uuid obtained using get_comp_list()
 * @return py::dict Dict containing component metatdata
 */
py::object get_comp_info(py::capsule& context, std::string& uuid) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_tid_t tid = uuid_to_tid(uuid);
  gxf_component_info_t info = new_component_info();

  gxf_result_t code = GxfComponentInfo(ctx, tid, &info);
  if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
    realloc_component_info(info, info.num_parameters);
    code = GxfComponentInfo(ctx, tid, &info);
  }
  if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }

  auto result = py::dict();
  result["typename"] = info.type_name;
  result["display_name"] = info.display_name;
  result["brief"] = info.brief;
  result["description"] = info.description;
  result["base_typename"] = info.base_name == nullptr ? "" : info.base_name;
  result["is_abstract"] = info.is_abstract ? true : false;
  destruct_component_info(info);

  return result;
}

/**
 * @brief Get the list of parameter keys in component specified using uuid
 *
 * @param context gxf context created using gxf_context_create()
 * @param uuid valid component uuid obtained using get_comp_list()
 * @return py::list List of parameter keys
 */
py::object get_param_list(py::capsule& context, std::string& uuid) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_tid_t tid = uuid_to_tid(uuid);
  gxf_component_info_t info = new_component_info();

  gxf_result_t code = GxfComponentInfo(ctx, tid, &info);
  if (code == GXF_QUERY_NOT_ENOUGH_CAPACITY) {
    realloc_component_info(info, info.num_parameters);
    code = GxfComponentInfo(ctx, tid, &info);
  }
  if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }

  auto result = py::list();
  for (uint i = 0; i < info.num_parameters; ++i) {
    std::string param_key(info.parameters[i]);
    result.append(param_key);
  }
  destruct_component_info(info);

  return result;
}

/**
 * @brief Get parameter specific metadata such as key, headline, description,
 * gxf_parameter_type, flags, handle_type, default_value
 *
 * @param context gxf context created using gxf_context_create()
 * @param uuid valid component uuid obtained using get_comp_list()
 * @param key valid parameter key obtained using get_param_list()
 * @return py::dict Dict containing parameter metatdata
 */
py::object get_param_info(py::capsule& context, std::string& uuid,
                        std::string& key) {
  gxf_context_t ctx = static_cast<gxf_context_t>(context);
  gxf_tid_t tid = uuid_to_tid(uuid);
  gxf_parameter_info_t info;

  gxf_result_t code = GxfParameterInfo(ctx, tid, key.c_str(), &info);
  if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }

  auto result = py::dict();
  result["key"] = info.key;
  result["headline"] = info.headline;
  result["description"] = info.description;
  result["gxf_parameter_type"] = GxfParameterTypeStr(info.type);
  result["rank"] = info.rank;
  auto shape = py::list();
  if (info.rank > 0){
      for (int i = 0; i < info.rank; ++i) {
        shape.append(info.shape[i]);
      }
  }
  else
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
    if (code != GXF_SUCCESS) {return pybind11::cast<pybind11::none>(Py_None); }

    result["handle_type"] = type_name;
  } else
    result["handle_type"] = "N/A";

    // Default value
  if(info.default_value) {
    if(info.type == GXF_PARAMETER_TYPE_INT8) {
      int8_t value = *static_cast<const int8_t*>(info.default_value);
      result["default"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
      int16_t value = *static_cast<const int16_t*>(info.default_value);
      result["default"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT32) {
      int32_t value = *static_cast<const int32_t*>(info.default_value);
      result["default"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_INT64) {
      int64_t value = *static_cast<const int64_t*>(info.default_value);
      result["default"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_UINT8) {
      uint8_t value = *static_cast<const uint8_t*>(info.default_value);
      result["default"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
      uint16_t value = *static_cast<const uint16_t*>(info.default_value);
      result["default"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_UINT32) {
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
  if(info.numeric_max) {
    if(info.type == GXF_PARAMETER_TYPE_UINT8) {
      uint8_t value = *static_cast<const uint8_t*>(info.numeric_max);
      result["max_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
      uint16_t value = *static_cast<const uint16_t*>(info.numeric_max);
      result["max_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_UINT32) {
      uint32_t value = *static_cast<const uint32_t*>(info.numeric_max);
      result["max_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
      uint64_t value = *static_cast<const uint64_t*>(info.numeric_max);
      result["max_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT8) {
      int8_t value = *static_cast<const int8_t*>(info.numeric_max);
      result["max_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
      int16_t value = *static_cast<const int16_t*>(info.numeric_max);
      result["max_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT32) {
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
  if(info.numeric_min) {
    if(info.type == GXF_PARAMETER_TYPE_UINT8) {
      uint8_t value = *static_cast<const uint8_t*>(info.numeric_min);
      result["min_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
      uint16_t value = *static_cast<const uint16_t*>(info.numeric_min);
      result["min_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_UINT32) {
      uint32_t value = *static_cast<const uint32_t*>(info.numeric_min);
      result["min_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
      uint64_t value = *static_cast<const uint64_t*>(info.numeric_min);
      result["min_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT8) {
      int8_t value = *static_cast<const int8_t*>(info.numeric_min);
      result["min_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
      int16_t value = *static_cast<const int16_t*>(info.numeric_min);
      result["min_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT32) {
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
  if(info.numeric_step) {
    if(info.type == GXF_PARAMETER_TYPE_UINT8) {
      uint8_t value = *static_cast<const uint8_t*>(info.numeric_step);
      result["step_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT16) {
      uint16_t value = *static_cast<const uint16_t*>(info.numeric_step);
      result["step_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_UINT32) {
      uint32_t value = *static_cast<const uint32_t*>(info.numeric_step);
      result["step_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_UINT64) {
      uint64_t value = *static_cast<const uint64_t*>(info.numeric_step);
      result["step_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT8) {
      int8_t value = *static_cast<const int8_t*>(info.numeric_step);
      result["step_value"] = value;
    } else if (info.type == GXF_PARAMETER_TYPE_INT16) {
      int16_t value = *static_cast<const int16_t*>(info.numeric_step);
      result["step_value"] = value;
    } else if(info.type == GXF_PARAMETER_TYPE_INT32) {
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
}

#endif  // PYGXF_HPP
