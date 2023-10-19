/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_ENTITY_WARDEN_HPP_
#define NVIDIA_GXF_STD_ENTITY_WARDEN_HPP_

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/component_factory.hpp"
#include "gxf/std/type_registry.hpp"

namespace nvidia {
namespace gxf {

// Takes care of all entities in the current application.
class EntityWarden {
 public:
  gxf_result_t create(gxf_uid_t eid);

  gxf_result_t initialize(gxf_uid_t eid);

  gxf_result_t deinitialize(gxf_uid_t eid);

  gxf_result_t destroy(gxf_uid_t eid, ComponentFactory* factory);

  gxf_result_t cleanup(ComponentFactory* factory);

  gxf_result_t isValid(gxf_uid_t eid);

  Expected<FixedVector<gxf_uid_t, kMaxEntities>> getAll();

  Expected<FixedVector<gxf_uid_t, kMaxComponents>> getEntityComponents(gxf_uid_t eid);

  gxf_result_t find(gxf_context_t context, const char* name, gxf_uid_t* eid);

  gxf_result_t addComponent(gxf_uid_t eid, gxf_uid_t cid, gxf_tid_t tid, void* raw_pointer,
                            Component* component);

  gxf_result_t addComponentToInterface(gxf_uid_t eid, gxf_uid_t cid, const char* name);

  Expected<gxf_uid_t> getComponentEntity(gxf_uid_t cid);

  Expected<gxf_tid_t> getComponentType(gxf_uid_t cid);

  gxf_result_t findComponent(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name,
                             int32_t* offset, TypeRegistry* type_registry, gxf_uid_t* cid);

  // Sets the mandatory parameter storage where parameters loaded from YAML are stored.
  void setParameterStorage(ParameterStorage* parameter_storage) {
    parameter_storage_ = parameter_storage;
  }

  // Warden need to create default EntityGroup at beginning
  // All created EntityItems are initialized with default EngityGroup uid
  gxf_result_t createDefaultEntityGroup(gxf_uid_t gid);

  // Create new EntityGroup. Caller provides group uid and name
  gxf_result_t createEntityGroup(gxf_uid_t gid, const char* name);

  // Update EntityGroup uid in EntityItem, to new group id gid
  // Remove eid from its previous EntityGroup, and add eid to its new group gid
  gxf_result_t updateEntityGroup(gxf_uid_t gid, gxf_uid_t eid);

  // Get all resource comonent cids from EntityGroup pointed by EntityItem eid
  Expected<FixedVector<gxf_uid_t, kMaxComponents>> getEntityGroupResources(gxf_uid_t eid);

  // Find all resource component cids place within entity eid
  // add cids into EntityGroup pointed by EntityItem
  gxf_result_t populateResourcesToEntityGroup(gxf_context_t context, gxf_uid_t eid);

  // Find all resource component cids place within entity eid
  // remove cids from EntityGroup pointed by EntityItem
  gxf_result_t depopulateResourcesFromEntityGroup(gxf_context_t context, gxf_uid_t eid);

  Expected<const char*> entityFindEntityGroupName(gxf_uid_t eid);

 private:
  enum struct Stage {
    kUninitialized = 0,
    kInitializationInProgress = 1,
    kInitialized = 2,
    kDeinitializationInProgress = 3,
    kDestructionInProgress = 4,
    kDestroyed = 5,
  };

  struct ComponentItem {
    gxf_uid_t cid;
    gxf_tid_t tid;
    void* raw_pointer;
    Component* component_pointer;
  };

  struct EntityItem {
    std::atomic<Stage> stage;
    gxf_uid_t uid;  // id of this struct itself
    FixedVector<ComponentItem, kMaxComponents> components;
    std::map<std::string, gxf_uid_t> interface;
    gxf_uid_t gid = kUnspecifiedUid;  // entity group id

    gxf_result_t initialize();

    gxf_result_t deinitialize();

    gxf_result_t destroy(ComponentFactory* factory);
  };

  // EntityGroupItem to group EntityItems to share some common
  // properties. All entities within the same EntityGroup has
  // visibility to all resource components in the group
  struct EntityGroupItem {
    gxf_uid_t gid = kUnspecifiedUid;
    std::string name;
    FixedVector<gxf_uid_t, kMaxEntities> entities;
    FixedVector<gxf_uid_t, kMaxComponents> resource_components;
  };

  std::recursive_mutex mutex_;
  std::map<gxf_uid_t, std::unique_ptr<EntityItem>> entities_;
  std::map<gxf_uid_t, std::unique_ptr<EntityGroupItem>> entity_groups_;
  gxf_uid_t default_entity_group_id_ = kUnspecifiedUid;
  ParameterStorage* parameter_storage_;

 private:
  // non-thread safe helper, need to be called within thread safe APIs
  gxf_result_t entityGroupRemoveEntity(gxf_uid_t eid);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_STD_ENTITY_WARDEN_HPP_
