/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_STD_TYPED_MESSAGE_VIEW_HPP_
#define NVIDIA_GXF_STD_TYPED_MESSAGE_VIEW_HPP_

#include <cstdarg>
#include <cstdint>
#include <type_traits>

#include "common/byte.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/memory_buffer.hpp"

namespace nvidia {
namespace gxf {

/// @brief Streamlines message creation

// TypedMessageViewInternal handles the recursive construction of the inner layers holding
// the types of interest and the names. It acts similar to a std::tuple, but in this case
// there is no "object" of the type, as is the case for std::tuple. The only thing saved
// is the type in comp_type_ and a const char* for the name (name to be used when adding
// to the entity). NOTE: the components are added to the entity in the function add_to_entity
// in reverse order as listed in the template. This is due to compile time constraints. See
// the function definition for example
template <typename... Ts>
struct TypedMessageViewInternal {
  virtual ~TypedMessageViewInternal() = default;

  TypedMessageViewInternal(){}
};  // base class, will end recursion

template <typename T, typename... Ts>
struct TypedMessageViewInternal<T, Ts...> : TypedMessageViewInternal<Ts...> {
  const char* component_name_;
  using comp_type_ = T;

  TypedMessageViewInternal() {}
  TypedMessageViewInternal(T t, Ts... ts) :
    TypedMessageViewInternal<Ts...>(ts...)
    {}

  virtual ~TypedMessageViewInternal() = default;
};

// base class for later template specialization
template <size_t, typename>
struct TypedMessageViewInternalHelper {};

template <typename T, typename... Ts>
struct TypedMessageViewInternalHelper<0, TypedMessageViewInternal<T, Ts...>> {
  using message_element_type_ = T;
  using element_type_ = TypedMessageViewInternal<T, Ts...>;
};

template <size_t I, typename T, typename... Ts>
struct TypedMessageViewInternalHelper<I, TypedMessageViewInternal<T, Ts...>>{
  using message_element_type_ = typename TypedMessageViewInternalHelper<I-1,
  TypedMessageViewInternal<Ts...>>::message_element_type_;
  using element_type_ = typename TypedMessageViewInternalHelper<I-1,
  TypedMessageViewInternal<Ts...>>::element_type_;
};

// get methods
template <size_t I, typename... Ts>
typename std::enable_if_t<I == 0,
                          typename TypedMessageViewInternalHelper<0,
                          TypedMessageViewInternal<Ts...>>::element_type_>
get_ele(TypedMessageViewInternal<Ts...> element) {
  return element;  // returns the slice of the struct, type TypedMessageViewInternal
}

template <size_t I, typename T, typename... Ts>
typename std::enable_if_t<I != 0,
                          typename TypedMessageViewInternalHelper<I,
                          TypedMessageViewInternal<T, Ts...>>::element_type_>
get_ele(TypedMessageViewInternal<T, Ts...> element) {
  TypedMessageViewInternal<Ts...> base = element;  // remove one slice
  return get_ele<I - 1>(base);
}

// get_name_rt is the runtime version of getting information, other methods are
// at compile time, which can't be used in the add_component_name scenario
template <typename T, typename... Ts>
const char*& get_name_rt(TypedMessageViewInternal<T, Ts...>& t, int k) {
  if (k > 1) {
    TypedMessageViewInternal<Ts...>& base = t;
    return get_name_rt(base, k - 1);
  } else if (k == 1) {
    TypedMessageViewInternal<Ts...>& base = t;
    return get_name_rt_final(base, k - 1);
  } else if (k < 0) {
    std::cout << "error in get_name_rt k < 1\n";
  }
  return t.component_name_;  // in this case k == 0, first element is already t
}

template <typename T>
const char*& get_name_rt(TypedMessageViewInternal<T>& t, int k) {
  return t.component_name_;
}

template <typename... Ts>
const char*& get_name_rt_final(TypedMessageViewInternal<Ts...>& t, int k) {
  if (k != 0) {
    std::cout << "error in get_name_rt_final\n";
  }
  return t.component_name_;
}

// add_component_helper recurses the TypedMessageView at compile time to add each saved type and
// corresponding name to the Entity
template<int N, typename T>
struct add_component_helper {
static Expected<void> f(T& t, Entity& entity) {
  auto result = entity.add<typename TypedMessageViewInternalHelper<N-1,
  T>::message_element_type_>(t.component_name_);
  GXF_ASSERT(result, "failed to add component");
  if (!result) { return ForwardError(result); }

  add_component_helper<N-1, T>::f(t, entity);  // recurse
  return Success;
}
};

// Specialized template
template<typename T>
struct add_component_helper<0, T> {
static Expected<void> f(T&, Entity&) {
  // Done recurse
  return Success;
}
};

// checks if the entity has all the types of the current slice of TypedMessageView
template<int N, typename T>
struct check_entity_helper {
static gxf_result_t f(T& t, Entity& entity) {
  auto result = entity.get<typename TypedMessageViewInternalHelper<N-1,
  T>::message_element_type_>(t.component_name_);
  if (!result) { return GXF_FAILURE; }

  check_entity_helper<N-1, T>::f(t, entity);  // recurse
  return GXF_SUCCESS;
}
};

// Specialized template
template<typename T>
struct check_entity_helper<0, T> {
static gxf_result_t f(T&, Entity&) {
  // Done recurse
  return GXF_SUCCESS;
}
};

// TypedMessageView is the user interface. It stores a name, a TypedMessageViewInternal,
// and a size. This needs to be constructed at compile time, thus the size must be
// known. Recursion is done with the above helper functions.
template <typename... Ts>
struct TypedMessageView{
  const char* name_ = "default_name";
  TypedMessageViewInternal<Ts...> format_ {};
  const size_t num_components = sizeof...(Ts);

  TypedMessageView(const char* name, ... ) {
    va_list argp;
    va_start(argp, name);
    add_component_names(name, argp);
  }

 public:
  TypedMessageViewInternal<Ts...>& get() {
    return format_;
  }


  // gxf_result_t add_component_name(const char* name, ...){
  gxf_result_t add_component_names(const char* name, va_list argp) {
    if (num_components < 1) {
      std::cout << "ERROR: add_component_name size error, must have at least one component\n";
      return GXF_ARGUMENT_INVALID;
    }

    get_name_rt(format_, 0) = name;  // add the first name to first place

    // loop over the number of types in format, each gets name
    for (size_t i = 1; i < num_components; ++i) {
      get_name_rt(format_, i) = va_arg(argp, const char*);
    }

    return GXF_SUCCESS;
  }

  // adds to entity in REVERSE ORDER AS LISTED
  // ie TypedMessageView<Tensor, double, int> ...
  // will be added to the Entity in the order:
  // int, then double, then Tensor.
  Expected<void> add_to_entity(Entity& entity) {
    const int size = sizeof...(Ts);
    add_component_helper<size, TypedMessageViewInternal<Ts...>>::f(format_, entity);
    // add_component_helper applies to all sections of the format (TypedMessageView)
    return Success;
  }


  gxf_result_t check_entity(Entity& entity) {
    const int size = sizeof...(Ts);
    gxf_result_t result = check_entity_helper<size,
    TypedMessageViewInternal<Ts...>>::f(format_, entity);  // executes for all slices of format
    return result;
  }
};

}  // namespace gxf
}  // namespace nvidia
#endif
