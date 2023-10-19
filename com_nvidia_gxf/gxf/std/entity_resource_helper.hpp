/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_GXF_STD_ENTITY_RESOURCE_FILTER_HPP_
#define NVIDIA_GXF_GXF_STD_ENTITY_RESOURCE_FILTER_HPP_

#include <vector>
#include "common/fixed_vector.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/resources.hpp"

namespace nvidia {
namespace gxf {

/// @brief A helper class to filter out resource components from an entity
class EntityResourceHelper {
 public:
  // Find all resource components, i.e. cub-class of ResourceBase, within entity eid
  // Return resource components cids
  static Expected<FixedVector<gxf_uid_t, kMaxComponents>>
    entityFindResources(gxf_context_t context, gxf_uid_t eid);


  // Return the ThreadPool grouped with entity eid, before which increment pool size
  // if 1) entity eid is specified as thread pinning entity
  // && 2) EntityGroup associated with entity eid has a ThreadPool resource
  // Note: if 2) is true but 1) is not true, still will not return the ThreadPool,
  // i.e. only return ThreadPool that is updated
  static Expected<Handle<ThreadPool>> updateAndGetThreadPool(gxf_context_t context, gxf_uid_t eid);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_GXF_STD_ENTITY_RESOURCE_FILTER_HPP_
