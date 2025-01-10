/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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
#pragma once

#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/byte.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/complex.hpp"
#include "gxf/std/memory_buffer.hpp"

namespace nvidia {
namespace gxf {

/**
 * @brief A wrapper around pybind11::object class that allows to be destroyed
 * with acquiring the GIL.
 *
 * This class is used in PyInputContext::py_receive() and PyOutputContext::py_emit() methods
 * to allow the Python code (decreasing the reference count) to be executed with the GIL acquired.
 *
 * Without this wrapper, the Python code would be executed without the GIL by the GXF execution
 * engine that destroys the Entity object and executes Message::~Message() and
 * pybind11::object::~object(), which would cause a segfault.
 */
#pragma GCC visibility push(hidden)
class GILGuardedPyObject {
 public:
  GILGuardedPyObject() = delete;
  explicit GILGuardedPyObject(const pybind11::object& obj) : obj_(obj) {}
  explicit GILGuardedPyObject(pybind11::object&& obj) : obj_(obj) {}

  pybind11::object& obj() { return obj_; }

  ~GILGuardedPyObject() {
    // Acquire GIL before destroying the PyObject
    pybind11::gil_scoped_acquire scope_guard;
    pybind11::handle handle = obj_.release();
    if (handle) { handle.dec_ref(); }
  }

 private:
  pybind11::object obj_;
};
#pragma GCC visibility pop

// A component which holds a Python pybind11:object. Multiple pybind11::objects can be added to one
// entity to create a map of pybind11::objects. The component name can be used as key.
class __attribute__((visibility("hidden"))) PyData {
 public:
  PyData() = default;

  ~PyData() {}

  PyData(const PyData&) = delete;

  PyData(PyData&& other) { *this = std::move(other); }

  PyData& operator=(const PyData&) = delete;

  PyData& operator=(PyData&& other) {
    data = std::move(other.data);
    return *this;
  }

  Expected<void> setData(pybind11::object& data) {
    pybind11::gil_scoped_acquire acquire;
    this->data = std::make_shared<nvidia::gxf::GILGuardedPyObject>(data);
    pybind11::gil_scoped_release release_gil;
    return Expected<void>();
  }

  Expected<pybind11::object> getData() {
    if (!this->data) {
      throw std::runtime_error("Data NULL");
      return Unexpected{GXF_ARGUMENT_NULL};
    }
    return this->data->obj();
  }

 private:
  std::shared_ptr<GILGuardedPyObject> data;
};

// Creates a new entity with a collection of named PyObjects Object
Expected<Entity> CreatePyDataMap(gxf_context_t context, std::initializer_list<PyData> descriptions,
                                 bool activate = true);

}  // namespace gxf
}  // namespace nvidia
