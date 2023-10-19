/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_CORE_RESOURCE_HPP_
#define NVIDIA_GXF_CORE_RESOURCE_HPP_

#include <memory>

#include "gxf/core/expected.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/resource_manager.hpp"

namespace nvidia {
namespace gxf {

template <typename T>
class Resource;

template <typename T>
class Resource<Handle<T>> {
 public:
  // Tries to get the Resource value. If first time call, query ResourceManager and save
  // the value for subsequent calls.
  // If invalid ResourceManager ptr, Handle<S>::Unspecified() keeps returning everytime
  // If ResourceManager cannot find the target, GXF_ENTITY_COMPONENT_NOT_FOUND
  // will be saved into value_. No repeat query.
  const Expected<Handle<T>>& try_get(const char* name = nullptr) const {
    if (value_ == Handle<T>::Unspecified()) {
      if (resource_manager_ == nullptr) {
        GXF_LOG_WARNING("Resource [type: %s] from comonent [cid: %ld] cannot get "
                        "its value because of nullptr ResourceManager",
                        TypenameAsString<T>(), owner_cid_);
        return unspecified_handle_;
      } else {
        Expected<Handle<T>> maybe_value =
          resource_manager_->findComponentResource<T>(owner_cid_, name);
        if (!maybe_value) {
          GXF_LOG_INFO("Resource [type: %s] from component [cid: %ld] "
                          "cannot find its value from ResourceManager",
                          TypenameAsString<T>(), owner_cid_);
          value_ = ForwardError(maybe_value);
          return value_;
        }
        value_ = maybe_value.value();
      }
    }
    return value_;
  }

  Expected<void> connect(std::shared_ptr<ResourceManager> resource_manager, gxf_uid_t owner_cid) {
    if (resource_manager == nullptr) {
      GXF_LOG_WARNING("nullptr ResourceManager ptr passed to connect Resource: %s from cid: %ld",
                      TypenameAsString<T>(), owner_cid);
    }
    resource_manager_ = resource_manager;
    owner_cid_ = owner_cid;
    return Success;
  }

 private:
  gxf_uid_t owner_cid_ = 0;
  std::shared_ptr<ResourceManager> resource_manager_;
  mutable Expected<Handle<T>> value_ = Handle<T>::Unspecified();
  // Used to return an Unexpected when the handle is not specified
  const Expected<Handle<T>> unspecified_handle_ = Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_CORE_RESOURCE_HPP_
