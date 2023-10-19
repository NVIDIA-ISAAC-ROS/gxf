/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/entity_resource_helper.hpp"
#include "gxf/std/entity_warden.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace nvidia {
namespace gxf {

gxf_result_t EntityWarden::create(gxf_uid_t eid) {
  // First create a new entity
  auto ptr = std::make_unique<EntityItem>();
  ptr->stage = Stage::kUninitialized;
  ptr->uid = eid;
  ptr->gid = default_entity_group_id_;

  // Then emplace it into the list under lock
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  entities_.emplace(eid, std::move(ptr));
  // TODO(byin): review to confirm: do not really add entities to default EntityGroup
  // But do add entities to user EntityGroup entity list besides assigning eg_id to entity item
  // entity_groups_[default_entity_group_id_]->entities.push_back(eid);

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::initialize(gxf_uid_t eid) {
  // Find the entity under lock.
  EntityItem* item;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    const auto it = entities_.find(eid);
    if (it == entities_.end()) {
      return GXF_ENTITY_NOT_FOUND;
    }
    item = it->second.get();

    for (const auto& comp : item->components) {
      auto result = parameter_storage_->isAvailable(comp.value().cid);
      if (!result) {
        return result.error();
      }
    }

    // To not allow initialization of an initialized object. initialize must be called only once.

    // Change the stage to pending under the lock
    if (item->stage != Stage::kUninitialized) {
      return GXF_INVALID_LIFECYCLE_STAGE;
    }
    item->stage = Stage::kInitializationInProgress;
  }

  // Then initialize the component outside of the lock.
  return item->initialize();
}

gxf_result_t EntityWarden::deinitialize(gxf_uid_t eid) {
  // Find the entity under lock.
  EntityItem* item;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    const auto it = entities_.find(eid);
    if (it == entities_.end()) {
      return GXF_ENTITY_NOT_FOUND;
    }
    item = it->second.get();

    // Allow deinitialization of uninitializes object as a no-op.
    if (item->stage == Stage::kUninitialized) {
      return GXF_SUCCESS;
    }

    // Change the stage to pending under the lock
    if (item->stage != Stage::kInitialized) {
      return GXF_INVALID_LIFECYCLE_STAGE;
    }
    item->stage = Stage::kDeinitializationInProgress;
  }

  // Then deinitialize the component outside of the lock.
  return item->deinitialize();
}

gxf_result_t EntityWarden::destroy(gxf_uid_t eid, ComponentFactory* factory) {
  // Find the entity under lock and remove it from the list.
  std::unique_ptr<EntityItem> item;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    const auto it = entities_.find(eid);
    if (it == entities_.end()) {
      return GXF_ENTITY_NOT_FOUND;
    }

    // To not allow destruction of an initialized object. Explicit call to deinitialize is
    // mandatory.

    // Remove the entity from the list and destroy it after
    item = std::move(it->second);
    entities_.erase(it);

    // Change the stage to pending under the lock
    if (item->stage != Stage::kUninitialized) {
      return GXF_INVALID_LIFECYCLE_STAGE;
    }
    item->stage = Stage::kDestructionInProgress;
  }

  // Destroy all components and the entity itself outside of the lock.
  return item->destroy(factory);
}

gxf_result_t EntityWarden::cleanup(ComponentFactory* factory) {
  gxf_result_t code = GXF_SUCCESS;

  // First move out all entities so that we can destroy them in peace. Note that calls to
  // deinitialize or destroy can make calls to this class.
  std::map<gxf_uid_t, std::unique_ptr<EntityItem>> tmp;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    tmp = std::move(entities_);
    entities_.clear();
  }

  // Deinitialize all
  for (auto& kvp : tmp) {
    EntityItem& item = *kvp.second;
    if (item.stage == Stage::kInitialized) {
      item.stage = Stage::kDeinitializationInProgress;
      const gxf_result_t current_code = item.deinitialize();
      code = AccumulateError(code, current_code);
    }
  }

  // Destroy all.
  for (auto& kvp : tmp) {
    EntityItem& item = *kvp.second;
    if (item.stage != Stage::kUninitialized) {
      code = GXF_INVALID_LIFECYCLE_STAGE;
    } else {
      item.stage = Stage::kDestructionInProgress;
      const gxf_result_t current_code = item.destroy(factory);
      code = AccumulateError(code, current_code);
    }
  }

  return code;
}

gxf_result_t EntityWarden::isValid(gxf_uid_t eid) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  const auto it = entities_.find(eid);
  return it == entities_.end() ? GXF_ENTITY_NOT_FOUND : GXF_SUCCESS;
}

Expected<FixedVector<gxf_uid_t, kMaxEntities>> EntityWarden::getAll() {
  FixedVector<gxf_uid_t, kMaxEntities> eids;
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  for (const auto& kvp : entities_) {
    auto result = eids.push_back(kvp.second->uid);
    if (!result) {
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  return eids;
}

Expected<FixedVector<gxf_uid_t, kMaxComponents>> EntityWarden::getEntityComponents(gxf_uid_t eid) {
  FixedVector<gxf_uid_t, kMaxComponents> cids;
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    return Unexpected{GXF_QUERY_NOT_FOUND};
  }
  const EntityItem* entity = it->second.get();

  for (const auto& component : entity->components) {
    auto result = cids.push_back(component.value().cid);
    if (!result) {
      GXF_LOG_ERROR("Current number of components in the entity is %ld while maximum"
                    " number of components allowed is %d", entity->components.size(),
                     kMaxComponents);
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  return cids;
}

gxf_result_t EntityWarden::find(gxf_context_t context, const char* name, gxf_uid_t* eid) {
  if (name == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (eid == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  std::unique_lock<std::recursive_mutex> lock(mutex_);

  const auto it =
      std::find_if(entities_.begin(), entities_.end(), [this, name, context](const auto& item) {
        const char* this_name = nullptr;
        const gxf_result_t result =
            GxfParameterGetStr(context, item.second->uid, kInternalNameParameterKey, &this_name);
        if (result != GXF_SUCCESS) {
          return false;
        }
        return std::strcmp(this_name, name) == 0;
      });

  if (it == entities_.end()) {
    *eid = kNullUid;
    return GXF_ENTITY_NOT_FOUND;
  } else {
    *eid = it->second->uid;
    return GXF_SUCCESS;
  }
}

gxf_result_t EntityWarden::addComponent(gxf_uid_t eid, gxf_uid_t cid, gxf_tid_t tid,
                                        void* raw_pointer, Component* component) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    return GXF_ENTITY_NOT_FOUND;
  }
  EntityItem& item = *it->second;

  if (item.stage != Stage::kUninitialized) {
    return GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION;
  }

  auto result = it->second->components.push_back({cid, tid, raw_pointer, component});
  if (!result) {
    switch (result.error()) {
      case FixedVectorBase<ComponentItem>::Error::kOutOfMemory: {
        return GXF_OUT_OF_MEMORY;
      } break;
      case FixedVectorBase<ComponentItem>::Error::kArgumentOutOfRange: {
        return GXF_ARGUMENT_OUT_OF_RANGE;
      } break;
      case FixedVectorBase<ComponentItem>::Error::kContainerEmpty: {
        return GXF_FAILURE;
      } break;
      case FixedVectorBase<ComponentItem>::Error::kContainerFull: {
        return GXF_OUT_OF_MEMORY;
      } break;
      default: {
        return GXF_FAILURE;
      } break;
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::addComponentToInterface(gxf_uid_t eid, gxf_uid_t cid,
                                                   const char* name) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    return GXF_ENTITY_NOT_FOUND;
  }
  EntityItem& item = *it->second;

  if (item.stage != Stage::kUninitialized) {
    return GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION;
  }
  item.interface.insert({std::string(name), cid});

  return GXF_SUCCESS;
}

Expected<gxf_uid_t> EntityWarden::getComponentEntity(gxf_uid_t cid) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  for (const auto& kvp : entities_) {
    for (const auto& ci : kvp.second->components) {
      if (ci.value().cid == cid) {
        return kvp.first;
      }
    }
  }
  return Unexpected{GXF_ENTITY_NOT_FOUND};
}

Expected<gxf_tid_t> EntityWarden::getComponentType(gxf_uid_t cid) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  for (const auto& kvp : entities_) {
    for (const auto& ci : kvp.second->components) {
      if (ci.value().cid == cid) {
        return ci.value().tid;
      }
    }
  }
  return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
}

gxf_result_t EntityWarden::findComponent(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid,
                                         const char* name, int32_t* offset,
                                         TypeRegistry* type_registry, gxf_uid_t* cid) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);

  const auto it = std::find_if(entities_.begin(), entities_.end(),
                               [eid](const auto& e) { return e.second->uid == eid; });
  if (it == entities_.end()) {
    return GXF_ENTITY_NOT_FOUND;
  }
  const EntityItem* entity = it->second.get();

  auto start = entity->components.begin();
  if (offset != nullptr) {
    if (*offset < 0) {
      GXF_LOG_ERROR("Offset cannot be negative %d", *offset);
      return GXF_FAILURE;
    }
    if (static_cast<size_t>(*offset) >= entity->components.size()) {
      return GXF_ENTITY_COMPONENT_NOT_FOUND;
    }
    std::advance(start, *offset);
  }

  const auto jt = std::find_if(start, entity->components.end(), [=](const auto& item) {
    if (GxfTidIsNull(tid) == 0) {
      const bool is_same_type = (*item).tid == tid;
      const bool is_derived_type = type_registry->is_base((*item).tid, tid);
      if (!is_same_type && !is_derived_type) {
        return false;
      }
    }
    if (name != nullptr) {
      const char* value = nullptr;
      const gxf_result_t code =
          GxfParameterGetStr(context, (*item).cid, kInternalNameParameterKey, &value);
      if (code != GXF_SUCCESS) {
        return false;
      }
      if (std::string(value) != std::string(name)) {
        return false;
      }
    }
    return true;
  });

  if (offset != nullptr) {
    *offset = std::distance(entity->components.begin(), jt);
  }

  // the component might be in the subgraph, try finding it from the interface
  // map
  if (jt == entity->components.end()) {
    if (name) {
      const auto i = entity->interface.find(std::string(name));
      if (i != entity->interface.end()) {
        *cid = i->second;
        return GXF_SUCCESS;
      }
    }
    return GXF_ENTITY_COMPONENT_NOT_FOUND;
  }

  *cid = (*jt).value().cid;
  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::EntityItem::initialize() {
  if (stage != Stage::kInitializationInProgress) {
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  for (size_t i = 0; i < components.size(); i++) {
    if (components[i].value().component_pointer == nullptr) {
      continue;
    }
    const gxf_result_t result = components[i].value().component_pointer->initialize();
    if (result != GXF_SUCCESS) {
      // Deinitialize components which were initialized so far
      for (size_t j = 0; j < i; j++) {
        if (components[j].value().component_pointer == nullptr) {
          continue;
        }
        const gxf_result_t result2 = components[j].value().component_pointer->deinitialize();
        // FIXME We can not propagate any errors.
        (void)result2;
      }
      stage = Stage::kUninitialized;
      GXF_LOG_ERROR("Failed to initialize component %05zu (%s)",
                    components[i].value().component_pointer->cid(),
                    components[i].value().component_pointer->name());
      return result;
    }
  }

  stage = Stage::kInitialized;

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::EntityItem::deinitialize() {
  if (stage != Stage::kDeinitializationInProgress) {
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  gxf_result_t code = GXF_SUCCESS;
  for (auto riter = components.rbegin(); riter != components.rend(); riter++) {
    Component* component = (*riter).value().component_pointer;
    if (component == nullptr) {
      continue;
    }
    const gxf_result_t result = component->deinitialize();
    if (result != GXF_SUCCESS) {
      const char* component_name =  "UNKNOWN";
      GxfComponentTypeName(component->context(), (*riter).value().tid, &component_name);
      GXF_LOG_WARNING("Component of type %s, cid %ld failed to deinitialize with code %s",
                      component_name, (*riter).value().cid, GxfResultStr(result));
    }
    code = AccumulateError(code, result);
  }

  stage = Stage::kUninitialized;

  return code;
}

gxf_result_t EntityWarden::EntityItem::destroy(ComponentFactory* factory) {
  if (factory == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (stage != Stage::kDestructionInProgress) {
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  gxf_result_t result = GXF_SUCCESS;
  for (const auto& component : components) {
    const Expected<void> code = factory->deallocate((*component).tid, (*component).raw_pointer);
    if (!code) {
      result = code.error();
    }
  }
  components.clear();

  stage = Stage::kDestroyed;

  return result;
}

gxf_result_t EntityWarden::createDefaultEntityGroup(gxf_uid_t gid) {
  const gxf_result_t result = this->createEntityGroup(gid, kDefaultEntityGroupName);
  if (result != GXF_SUCCESS) {
    return result;
  }
  default_entity_group_id_ = gid;
  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::createEntityGroup(gxf_uid_t gid, const char* name) {
  auto ptr = std::make_unique<EntityGroupItem>();
  ptr->gid = gid;
  if (name != nullptr) {
    ptr->name = std::string(name);
  }

  std::unique_lock<std::recursive_mutex> lock(mutex_);
  if (entity_groups_.find(gid) != entity_groups_.end()) {
    GXF_LOG_ERROR("EntityGroup with gid: %05zu already exists, "
                  "cannot create group using the same gid", gid);
    return GXF_FAILURE;
  }
  entity_groups_.emplace(gid, std::move(ptr));
  GXF_LOG_DEBUG("Created EntityGroup [gid: %05zu, name: %s]", gid, name);
  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::updateEntityGroup(gxf_uid_t gid, gxf_uid_t eid) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  // check if gid exists
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("EntityGroup with gid: %05zu is not created yet, "
                  "cannot add entity [eid: %05zu] into non-existant group", gid, eid);
    return GXF_ENTITY_GROUP_NOT_FOUND;
  }
  // check if eid exists
  const auto e_it = entities_.find(eid);
  if (e_it == entities_.end()) {
    GXF_LOG_ERROR("Cannot add non-existant entity [eid: %05zu] into EntityGroup [gid: %05zu]",
                  eid, gid);
    return GXF_ENTITY_NOT_FOUND;
  }
  // if both EntityGroup and entity are found, then
  // 0. record old gid
  gxf_uid_t gid_old = e_it->second->gid;
  if (gid_old == gid) {
    GXF_LOG_ERROR("Entity [eid: %05zu] was already added into EntityGroup [gid: %05zu]", eid, gid);
    return GXF_FAILURE;
  } else {
    if (gid_old == kUnspecifiedUid) {
      GXF_LOG_ERROR("Entity [eid: %05zu] is not intialized to default EntityGroup", eid);
      return GXF_FAILURE;
    } else if (gid_old == default_entity_group_id_) {
      GXF_LOG_DEBUG(
        "Entity [eid: %05zu] switching from default EntityGroup [gid: %05zu] "
        "to user's [gid: %05zu]", eid, gid_old, gid);
    } else {
      GXF_LOG_DEBUG(
        "Entity [eid: %05zu] overwriting user EntityGroup from [gid: %05zu] to [gid: %05zu]",
        eid, gid_old, gid);
    }
  }

  // 1. remove entity id from old EntityGroup
  gxf_result_t rm_result = this->entityGroupRemoveEntity(eid);
  if (rm_result != GXF_SUCCESS) { return rm_result; }

  // 2. update new EntityGroup id in the entity
  // EntityItem::gid
  e_it->second->gid = gid;

  // 3. add entity id into new EntityGroup
  // EntityGroupItem::entities
  auto result = eg_it->second->entities.push_back({eid});
  if (!result) {
    switch (result.error()) {
      case FixedVectorBase<gxf_uid_t>::Error::kOutOfMemory: {
        return GXF_OUT_OF_MEMORY;
      } break;
      case FixedVectorBase<gxf_uid_t>::Error::kArgumentOutOfRange: {
        return GXF_ARGUMENT_OUT_OF_RANGE;
      } break;
      case FixedVectorBase<gxf_uid_t>::Error::kContainerEmpty: {
        return GXF_FAILURE;
      } break;
      case FixedVectorBase<gxf_uid_t>::Error::kContainerFull: {
        return GXF_OUT_OF_MEMORY;
      } break;
      default: {
        return GXF_FAILURE;
      } break;
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::entityGroupRemoveEntity(gxf_uid_t eid) {
  // non-thread safe private helper
  // check if eid exists
  const auto e_it = entities_.find(eid);
  if (e_it == entities_.end()) {
    GXF_LOG_ERROR("Cannot remove non-existant entity [eid: %05zu] from its EntityGroup", eid);
    return GXF_ENTITY_NOT_FOUND;
  }
  // check if gid exists
  gxf_uid_t gid = e_it->second->gid;
  if (gid == kUnspecifiedUid) {
    GXF_LOG_ERROR("Entity [eid: %05zu] already has no EntityGroup", eid);
    return GXF_FAILURE;
  }
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existant EntityGroup [gid: %05zu]", eid, gid);
    return GXF_ENTITY_GROUP_NOT_FOUND;
  }

  // 1. remove entity id from pointed EntityGroup
  // caution: erase by index while iterating
  for (size_t i = 0; i < eg_it->second->entities.size(); i++) {
    if (eg_it->second->entities.at(i).value() == eid) {
      eg_it->second->entities.erase(i);
      GXF_LOG_DEBUG("EntityGroup [gid: %05zu] removed entity [eid: %05zu]", gid, eid);
    }
  }
  // 2. unset entity's EntityGroup id
  e_it->second->gid = kUnspecifiedUid;
  return GXF_SUCCESS;
}

Expected<FixedVector<gxf_uid_t, kMaxComponents>>
  EntityWarden::getEntityGroupResources(gxf_uid_t eid) {
  FixedVector<gxf_uid_t, kMaxComponents> resource_cids;
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  // check if eid exists
  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    GXF_LOG_ERROR("Cannot find entity [eid: %05zu]", eid);
    return Unexpected{GXF_ENTITY_NOT_FOUND};
  }
  // check if gid exists
  gxf_uid_t gid = it->second->gid;
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existant EntityGroup [gid: %05zu]", eid, gid);
    return Unexpected{GXF_ENTITY_GROUP_NOT_FOUND};
  }

  // collect all resource comonents from EntityGroup pointed by eid
  for (size_t i = 0; i < eg_it->second->resource_components.size(); i++) {
    resource_cids.push_back(eg_it->second->resource_components.at(i).value());
  }
  return resource_cids;
}

gxf_result_t EntityWarden::populateResourcesToEntityGroup(gxf_context_t context, gxf_uid_t eid) {
  // find all resources placed within entity eid
  auto maybe_resources = EntityResourceHelper::entityFindResources(context, eid);
  if (!maybe_resources) { return ToResultCode(maybe_resources); }
  auto resources = maybe_resources.value();
  if (resources.empty()) {
    return GXF_SUCCESS;
  }

  // lock
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  // check if eid exists
  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    GXF_LOG_ERROR("Cannot find entity [eid: %05zu]", eid);
    return GXF_ENTITY_NOT_FOUND;
  }
  // check if gid exists
  gxf_uid_t gid = it->second->gid;
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existant EntityGroup [gid: %05zu]", eid, gid);
    return GXF_ENTITY_GROUP_NOT_FOUND;
  }

  // add all resource cids into EntityGroup pointed by entity eid
  for (size_t i = 0; i < resources.size(); i++) {
    eg_it->second->resource_components.push_back(resources.at(i).value());
  }
  return GXF_SUCCESS;
}

// Assumption: no new resource components are added after entity get activated
gxf_result_t EntityWarden::depopulateResourcesFromEntityGroup(gxf_context_t context,
                                                              gxf_uid_t eid) {
  // find all resources placed within entity eid
  auto maybe_resources = EntityResourceHelper::entityFindResources(context, eid);
  if (!maybe_resources) { return ToResultCode(maybe_resources); }
  auto resources = maybe_resources.value();
  if (resources.empty()) {
    return GXF_SUCCESS;
  }

  // lock
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  // check if eid exists
  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    GXF_LOG_ERROR("Cannot find entity [eid: %05zu]", eid);
    return GXF_ENTITY_NOT_FOUND;
  }
  // check if gid exists
  gxf_uid_t gid = it->second->gid;
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existant EntityGroup [gid: %05zu]", eid, gid);
    return GXF_ENTITY_GROUP_NOT_FOUND;
  }

  // 1. remove resource component cids from pointed EntityGroup
  // caution: erase by index while iterating
  for (size_t i = 0; i < eg_it->second->resource_components.size(); i++) {
    for (size_t j = 0; j < resources.size(); j++) {
      if (eg_it->second->resource_components.at(i).value() == resources.at(j).value()) {
        eg_it->second->resource_components.erase(i);
        GXF_LOG_DEBUG(
          "EntityGroup [gid: %05zu] removed resource [cid: %05zu]", gid, resources.at(j).value());
      }
    }
  }

  return GXF_SUCCESS;
}

Expected<const char*> EntityWarden::entityFindEntityGroupName(gxf_uid_t eid) {
  // check if eid exists
  const auto e_it = entities_.find(eid);
  if (e_it == entities_.end()) {
    GXF_LOG_ERROR("Non-existant entity [eid: %05zu]", eid);
    return Unexpected{GXF_ENTITY_NOT_FOUND};
  }
  // check if gid exists
  gxf_uid_t gid = e_it->second->gid;
  if (gid == kUnspecifiedUid) {
    GXF_LOG_ERROR("Entity [eid: %05zu] has no EntityGroup", eid);
    return Unexpected{GXF_FAILURE};
  }
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existant EntityGroup [gid: %05zu]", eid, gid);
    return Unexpected{GXF_ENTITY_GROUP_NOT_FOUND};
  }
  return eg_it->second->name.c_str();
}

}  // namespace gxf
}  // namespace nvidia
