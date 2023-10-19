/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

#include "gxf/python_codelet/py_codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t PyCodeletV0::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(codelet_name_, "codelet_name", "Codelet Name",
                                 "Name of the python codelet");
  result &= registrar->parameter(codelet_filepath_, "codelet_file", "Absolute Codelet File Path",
                                 "Absolute path to the file containing the codelet implementation");
  result &= registrar->parameter(codelet_params_, "codelet_params", "Params", "Codelet params",
                                 Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t PyCodeletV0::start() {
  // acquire GIL
  pybind11::gil_scoped_acquire acquire;
  std::string codelet_filepath = codelet_filepath_.get();
  std::string codelet_dir;
  std::string codelet_filename;

  if (codelet_filepath.empty() || codelet_filepath == "None") {
    try {
      pybind11::module::import("sys").attr("path").attr("append")(codelet_dir.c_str());
      pybind11::module pycodelet_module = pybind11::module::import("__main__");
      pycodelet = pycodelet_module.attr(codelet_name_.get().c_str())();
    } catch (std::exception& e) {
      GXF_LOG_ERROR("%s", e.what());
      return GXF_FAILURE;
    }
  } else {
    const size_t pos = codelet_filepath.find_last_of('/');
    if (pos == std::string::npos) {
      GXF_LOG_WARNING("[E%05zu] Please provide absolute path to the python codelet: %s",
                      this->eid(), codelet_filepath.c_str());
      codelet_dir = "";
      codelet_filename = codelet_filepath;
    } else {
      codelet_dir = codelet_filepath.substr(0, pos);
      codelet_filename = codelet_filepath.substr(pos + 1);
    }
    std::string codelet_module_name;
    const size_t pos2 = codelet_filename.find('.');

    if (pos2 == std::string::npos) {
      GXF_LOG_ERROR("[E%05zu] File : %s should be a python file ending in .py", this->eid(),
                    codelet_filename.c_str());
      return GXF_FAILURE;
    } else {
      codelet_module_name = codelet_filename.substr(0, pos2);
    }

    try {
      pybind11::module::import("sys").attr("path").attr("append")(codelet_dir.c_str());
      pybind11::module pycodelet_module = pybind11::module::import(codelet_module_name.c_str());
      pycodelet = pycodelet_module.attr(codelet_name_.get().c_str())();
    } catch (std::exception& e) {
      GXF_LOG_ERROR("%s", e.what());
      return GXF_FAILURE;
    }
  }

  try {
    pycodelet.attr("set_bridge")(this);
    pycodelet.attr("start")();
  } catch (const std::exception& e) {
    GXF_LOG_ERROR("%s", e.what());
    return GXF_FAILURE;
  }

  // release GIL
  pybind11::gil_scoped_release release_gil;
  return GXF_SUCCESS;
}

gxf_result_t PyCodeletV0::tick() {
  pybind11::gil_scoped_acquire acquire;
  try {
    pycodelet.attr("tick")();
  } catch (std::exception& e) {
    GXF_LOG_ERROR("%s", e.what());
    return GXF_FAILURE;
  }
  pybind11::gil_scoped_release release_gil;
  return GXF_SUCCESS;
}

gxf_result_t PyCodeletV0::stop() {
  pybind11::gil_scoped_acquire acquire;
  try {
    pycodelet.attr("stop")();
  } catch (std::exception& e) {
    GXF_LOG_ERROR("%s", e.what());
    return GXF_FAILURE;
  }
  pybind11::gil_scoped_release release_gil;
  return GXF_SUCCESS;
}

std::string PyCodeletV0::getParams() {
  auto maybe_codelet_params = codelet_params_.try_get();
  if (!maybe_codelet_params) {
    GXF_LOG_ERROR("[E%05zu] Couldn't get codelet params", this->eid());
    throw std::runtime_error(GxfResultStr(GXF_PARAMETER_NOT_INITIALIZED));
  }
  return maybe_codelet_params.value();
}

}  // namespace gxf
}  // namespace nvidia
