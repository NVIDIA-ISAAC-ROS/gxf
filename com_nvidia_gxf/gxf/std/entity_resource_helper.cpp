/*
Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <utility>
#include <vector>

#include "gxf/core/resource_manager.hpp"
#include "gxf/std/cpu_thread.hpp"
#include "gxf/std/entity_resource_helper.hpp"
#include "gxf/std/resources.hpp"

namespace nvidia {
namespace gxf {
Expected<FixedVector<gxf_uid_t, kMaxComponents>>
  EntityResourceHelper::entityFindResources(gxf_context_t context, gxf_uid_t eid) {
  FixedVector<gxf_uid_t, kMaxComponents> resource_cids;
  // Get handle to entity
  auto maybe = Entity::Shared(context, eid);
  if (!maybe) { return ForwardError(maybe); }
  auto entity = maybe.value();
  // Find all Resources components
  auto maybe_resources = entity.findAllHeap<ResourceBase>();
  if (!maybe_resources) { return Unexpected{GXF_FAILURE}; }
  auto resources = std::move(maybe_resources.value());
  // look for resource components from current entity
  for (size_t i = 0; i < resources.size(); i++) {
    if (!resources.at(i)) {
      GXF_LOG_ERROR("Invalid Resource");
      return Unexpected{GXF_FAILURE};
    }
    resource_cids.push_back(resources.at(i).value().cid());
    GXF_LOG_DEBUG("Find resource [cid: %05zu, name: %s] from entity [eid: %05zu, name: %s]",
                  resources.at(i).value().cid(), resources.at(i).value().name(),
                  entity.eid(), entity.name());
  }
  return resource_cids;
}

Expected<Handle<ThreadPool>>
  EntityResourceHelper::updateAndGetThreadPool(gxf_context_t context, gxf_uid_t eid) {
  bool need_update = false;
  auto maybe = Entity::Shared(context, eid);
  if (!maybe) { return ForwardError(maybe); }
  auto entity = maybe.value();
  // Find all Resources components
  auto maybe_cpu_thread = entity.findAllHeap<CPUThread>();
  if (!maybe_cpu_thread) { return Unexpected{GXF_FAILURE}; }
  auto cpu_thread = std::move(maybe_cpu_thread.value());
  if (!cpu_thread.empty()) {
    if (cpu_thread.size() > 1) {
      GXF_LOG_ERROR("More than one CPUThread (%lu) added "
                    "into the same Entity [eid: %05zu, name: %s]",
                    cpu_thread.size(), eid, entity.name());
      return Unexpected{GXF_FAILURE};
    }
    if (cpu_thread.at(0).value()->pinned()) {
      need_update = true;
    }
  }
  if (!need_update) {
    GXF_LOG_DEBUG("Entity [eid: %05zu, name: %s] is not marked as thread pinnings",
                  eid, entity.name());
    return Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  // return the first ThreadPool
  auto maybe_thread_pool = ResourceManager::findEntityResource<ThreadPool>(context, eid);
  if (!maybe_thread_pool) {
    GXF_LOG_ERROR("CPUThread pinned Entity[eid: %05zu, name: %s] doesn't have a ThreadPool",
                  eid, entity.name());
    return Unexpected{GXF_RESOURCE_NOT_FOUND};
  }
  auto thread_pool = maybe_thread_pool.value();
  auto maybe_added = thread_pool->addThread(eid);
  if (!maybe_added) {
    GXF_LOG_ERROR(
      "ThreadPool [cid: %05zu, name: %s] failed to add thread for pinned entity "
      "[eid: %05zu, name: %s]", thread_pool->cid(), thread_pool->name(), eid, entity.name());
  } else {
    GXF_LOG_DEBUG(
      "ThreadPool [cid: %05zu, name: %s] created thread [uid: %05zu] for pinned entity "
      "[eid: %05zu, name: %s]", thread_pool->cid(), thread_pool->name(),
      maybe_added.value(), eid, entity.name());
  }
  return maybe_thread_pool;
}

}  // namespace gxf
}  // namespace nvidia
