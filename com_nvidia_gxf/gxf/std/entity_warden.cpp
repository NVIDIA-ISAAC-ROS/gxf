/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace nvidia {
namespace gxf {

gxf_result_t EntityWarden::create(gxf_uid_t eid, EntityItem** item_ptr,
                                  const std::string& entity_name) {
  // First create a new entity
  auto ptr = std::make_unique<EntityItem>();
  ptr->stage = Stage::kUninitialized;
  ptr->uid = eid;
  ptr->gid = default_entity_group_id_;
  if (item_ptr != nullptr) {
    *item_ptr = ptr.get();
  }

  {
    std::unique_lock<std::shared_mutex> names_lock(names_mutex_);
    eid_to_name_.emplace(eid, entity_name);
    name_to_eid_.emplace(entity_name, eid);
  }
  std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
  // Then emplace it into the list under lock
  entities_.emplace(eid, std::move(ptr));
  // TODO(byin): review to confirm: do not really add entities to default EntityGroup
  // But do add entities to user EntityGroup entity list besides assigning eg_id to entity item
  // entity_groups_[default_entity_group_id_]->entities.push_back(eid);

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::findUninitialized(gxf_uid_t eid, EntityItem *& item) const {
  const auto it = entities_.find(eid);

  if (it == entities_.end()) {
    return GXF_ENTITY_NOT_FOUND;
  }

  item = it->second.get();

  std::shared_lock<std::shared_mutex> lock_(item->entity_item_mutex_);
  for (const auto& comp : item->components) {
    auto result = parameter_storage_->isAvailable(comp.value().cid);
    if (!result) {
      return result.error();
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::initialize(gxf_uid_t eid) {
  // Find the entity under read lock.
  EntityItem* item;
  {
    std::unique_lock<std::shared_mutex> entity_item_lock;
    {
      std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);
      gxf_result_t find_result = findUninitialized(eid, item);
      if (find_result != GXF_SUCCESS) {
        return find_result;
      }

      entity_item_lock = std::unique_lock<std::shared_mutex>(item->entity_item_mutex_);
    }
    // To not allow initialization of an initialized
    // object. Initialize must be called only once.
    //
    // This check must happen under the write lock to avoid race
    // conditions.
    if (item->stage != Stage::kUninitialized) {
      return GXF_INVALID_LIFECYCLE_STAGE;
    }

    // Change the stage to pending under the entity item lock
    item->stage = Stage::kInitializationInProgress;
  }

  // Then initialize the component outside of the locks.
  return item->initialize();
}

gxf_result_t EntityWarden::deinitialize(gxf_uid_t eid) {
  // Find the entity under lock.
  EntityItem* item;
  {
    {
      // Acquire shared lock on entities_
      std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);
      const auto it = entities_.find(eid);
      if (it == entities_.end()) {
        return GXF_ENTITY_NOT_FOUND;
      }
      item = it->second.get();
      // Release shared lock on entities_
    }

    // Acquire unique lock on entity item.
    auto item_lock = std::unique_lock<std::shared_mutex>(item->entity_item_mutex_);

    // Allow deinitialization of uninitializes object as a no-op.
    if (item->stage == Stage::kUninitialized) {
      return GXF_SUCCESS;
    }

    // Change the stage to pending under the lock
    if (item->stage != Stage::kInitialized) {
      return GXF_INVALID_LIFECYCLE_STAGE;
    }
    item->stage = Stage::kDeinitializationInProgress;
    // Release unique lock on entity item
  }

  // Then deinitialize the component outside of the locks.
  return item->deinitialize();
}

gxf_result_t EntityWarden::destroy(gxf_uid_t eid, ComponentFactory* factory) {
  // Find the entity under lock and remove it from the list.
  std::unique_ptr<EntityItem> item;
  {  // Acquire unique lock for entities_ and components_
    std::unique_lock<std::shared_mutex> entity_item_lock;
    {
      std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
      const auto it = entities_.find(eid);
      if (it == entities_.end()) { return GXF_ENTITY_NOT_FOUND; }

      // To not allow destruction of an initialized object. Explicit call to deinitialize is
      // mandatory.

      // Remove the entity from the list and destroy it after
      item = std::move(it->second);
      entities_.erase(it);
      // Acquire unique lock for item
      entity_item_lock = std::unique_lock<std::shared_mutex>(item->entity_item_mutex_);
      for (const auto& component : item->components) {
        auto iter = components_.find(component->cid);
        if (iter != components_.end()) {
          components_.erase(iter);
        }
      }
    }
    {
      std::unique_lock<std::shared_mutex> name_lock(names_mutex_);
      name_to_eid_.erase(eid_to_name_[eid]);
      eid_to_name_.erase(eid);
    }
    // Change the stage to pending under the lock
    if (item->stage != Stage::kUninitialized) {
      return GXF_INVALID_LIFECYCLE_STAGE;
    }
    item->stage = Stage::kDestructionInProgress;
    // Release unique lock on item
  }

  // Destroy all components and the entity itself outside of the lock.
  return item->destroy(factory);
}

gxf_result_t EntityWarden::cleanup(ComponentFactory* factory) {
  gxf_result_t code = GXF_SUCCESS;

  // First move out all entities so that we can destroy them in peace. Note that calls to
  // deinitialize or destroy can make calls to this class.
  std::unordered_map<gxf_uid_t, std::unique_ptr<EntityItem>> tmp;
  {
    std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
    tmp = std::move(entities_);
    entities_.clear();
    components_.clear();
  }
  {
    std::unique_lock<std::shared_mutex> lock(names_mutex_);
    name_to_eid_.clear();
    eid_to_name_.clear();
  }

  // Deinitialize all
  for (auto& kvp : tmp) {
    EntityItem& item = *kvp.second;
    if (item.stage == Stage::kInitialized) {
      std::unique_lock<std::shared_mutex> item_lock(item.entity_item_mutex_);
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
      std::unique_lock<std::shared_mutex> item_lock(item.entity_item_mutex_);
      item.stage = Stage::kDestructionInProgress;
      const gxf_result_t current_code = item.destroy(factory);
      code = AccumulateError(code, current_code);
    }
  }

  return code;
}

gxf_result_t EntityWarden::isValid(gxf_uid_t eid) const {
  std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);
  const auto it = entities_.find(eid);
  return it == entities_.end() ? GXF_ENTITY_NOT_FOUND : GXF_SUCCESS;
}

Expected<FixedVector<gxf_uid_t, kMaxEntities>> EntityWarden::getAll() const {
  FixedVector<gxf_uid_t, kMaxEntities> eids;
  std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);

  for (const auto& kvp : entities_) {
    auto result = eids.push_back(kvp.second->uid);
    if (!result) {
      return Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  return eids;
}

Expected<FixedVector<gxf_uid_t, kMaxComponents>> EntityWarden::getEntityComponents(
    gxf_uid_t eid) const {
  FixedVector<gxf_uid_t, kMaxComponents> cids;
  std::shared_lock<std::shared_mutex> entity_item_lock;
  const EntityItem* entity;
  {
    std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);
    const auto it = entities_.find(eid);
    if (it == entities_.end()) {
      return Unexpected{GXF_QUERY_NOT_FOUND};
    }
    entity_item_lock = std::shared_lock<std::shared_mutex>(it->second.get()->entity_item_mutex_);
    entity = it->second.get();
  }

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

gxf_result_t EntityWarden::find(gxf_context_t context, const char* name, gxf_uid_t* eid) const {
  if (name == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  if (eid == nullptr) {
    return GXF_ARGUMENT_NULL;
  }

  if (name[0] == '\0') {
    *eid = kNullUid;
    return GXF_ENTITY_NOT_FOUND;
  }
  std::shared_lock<std::shared_mutex> lock(names_mutex_);
  auto itr = name_to_eid_.find(name);
  if (itr == name_to_eid_.end()) {
    *eid = kNullUid;
    return GXF_ENTITY_NOT_FOUND;
  } else {
    *eid = itr->second;
    return GXF_SUCCESS;
  }
}

gxf_result_t EntityWarden::removeComponent(gxf_context_t context, gxf_uid_t eid,
                                           gxf_uid_t cid, ComponentFactory * factory) {
    if (factory == nullptr) {  return GXF_ARGUMENT_NULL; }
    EntityItem* item;
    std::unique_lock<std::shared_mutex> entity_item_lock;
    {
      // Acquiring unique lock on components_ and entities_
      std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
      auto itr = components_.find(cid);
      if (itr == components_.end()) {
        GXF_LOG_ERROR("Invalid component id %lu.", cid);
        return GXF_ENTITY_COMPONENT_NOT_FOUND;
      }
      components_.erase(itr);
      const auto it = entities_.find(eid);
      if (it == entities_.end()) {
        GXF_LOG_ERROR("Entity with uid %lu not found.", eid);
        return GXF_ENTITY_NOT_FOUND;
      }
      item = it->second.get();
      // Acquiring unique lock on entity item
      entity_item_lock = std::unique_lock<std::shared_mutex>(item->entity_item_mutex_);
      // Releasing unique lock on components_ and entities_
    }
    if (item->stage != Stage::kUninitialized) {
      return GXF_ENTITY_CAN_NOT_REMOVE_COMPONENT_AFTER_INITIALIZATION;
    }
    for (size_t i = 0; i < item->components.size(); i++) {
      if (cid == item->components[i].value().cid) {
        const Expected<void> code = factory->deallocate(item->components[i].value().tid,
         item->components[i].value().component_pointer);
        if (!code) {
          return code.error();
        }
        item->components.erase(i);
        return GXF_SUCCESS;
      }
    }
    // Releasing unique lock on entity item
    return GXF_SUCCESS;
}

gxf_result_t EntityWarden::addComponent(gxf_uid_t eid, gxf_uid_t cid, gxf_tid_t tid,
                                        void* raw_pointer, Component* component) {
  {
    EntityItem* item;
    std::unique_lock<std::shared_mutex> entity_item_lock;
    { // Acquire unique lock on entities_
      std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);

      const auto it = entities_.find(eid);
      if (it == entities_.end()) {
        return GXF_ENTITY_NOT_FOUND;
      }
      components_[cid] = ComponentEntityType{eid, tid};
      item = it->second.get();
      // Acquire unique lock on entity item
      entity_item_lock = std::unique_lock<std::shared_mutex>(item->entity_item_mutex_);
      // Release unique lock on entities_
    }
    if (item->stage != Stage::kUninitialized) {
      return GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION;
    }

    auto result = item->components.push_back({cid, tid, raw_pointer, component});
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
          return GXF_ENTITY_MAX_COMPONENTS_LIMIT_EXCEEDED;
        } break;
        default: {
          return GXF_FAILURE;
        } break;
      }
    }
    // Release unique lock on entity item
  }

  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::addComponentToInterface(gxf_uid_t eid, gxf_uid_t cid,
                                                   const char* name) {
  std::unique_lock<std::shared_mutex> entity_item_lock;
  EntityItem * item;
  {
    // Acquire shared lock on entities_
    std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);

    const auto it = entities_.find(eid);
    if (it == entities_.end()) {
      return GXF_ENTITY_NOT_FOUND;
    }
    item = it->second.get();
    // Acquire unique lock on entity item
    entity_item_lock = std::unique_lock<std::shared_mutex>(item->entity_item_mutex_);
    // Release shared lock on entities_
  }
  if (item->stage != Stage::kUninitialized) {
    return GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION;
  }
  item->interface.insert({std::string(name), cid});

  // Released unique lock on entity item
  return GXF_SUCCESS;
}

Expected<gxf_uid_t> EntityWarden::getComponentEntity(gxf_uid_t cid) const {
  std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);

  auto iter = components_.find(cid);
  if (iter != components_.end()) {
    return iter->second.eid;
  }

  return Unexpected{GXF_ENTITY_NOT_FOUND};
}

Expected<EntityItem*> EntityWarden::getEntityPtr(gxf_uid_t eid) const {
  std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);
  auto iter = entities_.find(eid);
  if (iter != entities_.end()) {
    return (iter->second).get();
  }
  return Unexpected{GXF_ENTITY_NOT_FOUND};
}

gxf_result_t EntityWarden::getEntityName(gxf_uid_t eid, const char** entity_name) const {
  if (entity_name == nullptr) { return GXF_ARGUMENT_NULL; }
  std::shared_lock<std::shared_mutex> lock(names_mutex_);
  auto iter = eid_to_name_.find(eid);
  if (iter != eid_to_name_.end()) {
    *entity_name = iter->second.c_str();
    return GXF_SUCCESS;
  }
  return GXF_ENTITY_NOT_FOUND;
}

Expected<gxf_tid_t> EntityWarden::getComponentType(gxf_uid_t cid) const {
  std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);

  auto iter = components_.find(cid);
  if (iter != components_.end()) {
    return iter->second.tid;
  }

  return Unexpected{GXF_ENTITY_COMPONENT_NOT_FOUND};
}

gxf_result_t EntityWarden::findComponent(gxf_context_t context, EntityItem* entity, gxf_tid_t tid,
                                         const char* name, int32_t* offset,
                                         TypeRegistry* type_registry, gxf_uid_t* cid,
                                         void** comp_ptr) const {
  std::shared_lock<std::shared_mutex> entity_item_lock(entity->entity_item_mutex_);
  auto eid = entity->uid;
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

  const bool is_tid_valid = (GxfTidIsNull(tid) == 0);
  const auto jt = std::find_if(start, entity->components.end(), [=](const auto& item) {
    if (is_tid_valid) {
      const bool is_same_type = (*item).tid == tid;
      if (is_same_type == false) {
        auto base_result = type_registry->is_base((*item).tid, tid);
        if (!base_result) { return false; }
        const bool is_derived_type = base_result.value();
        if (is_derived_type == false) {
          return false;
        }
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
        if (isSuccessful(GxfComponentPointer(context, *cid, tid, comp_ptr))) {
          return GXF_SUCCESS;
        } else {
          GXF_LOG_ERROR("Could not find component pointer from the interface map for entity %lu",
           eid);
          return GXF_ENTITY_COMPONENT_NOT_FOUND;
        }
      }
    }
    return GXF_ENTITY_COMPONENT_NOT_FOUND;
  }

  *cid = (*jt).value().cid;
  *comp_ptr = (*jt).value().component_pointer;
  // Release shared lock on entity item
  return GXF_SUCCESS;
}

gxf_result_t EntityItem::initialize() {
  if (stage != Stage::kInitializationInProgress) {
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  for (size_t i = 0; i < components.size(); i++) {
    if (components[i].value().component_pointer == nullptr) {
      continue;
    }
    const auto result = static_cast<Component*>(components[i].value()
    .component_pointer)->initialize();
    if (result != GXF_SUCCESS) {
      // Deinitialize components which were initialized so far
      for (size_t j = 0; j < i; j++) {
        if (components[j].value().component_pointer == nullptr) {
          continue;
        }
        const auto result2 = static_cast<Component*>(components[j].value().
        component_pointer)->deinitialize();
        // FIXME We can not propagate any errors.
        (void)result2;
      }
      stage = Stage::kUninitialized;
      GXF_LOG_ERROR("Failed to initialize component %05zu (%s)",
                    static_cast<Component*>(components[i].value().component_pointer)->cid(),
                    static_cast<Component*>(components[i].value().component_pointer)->name());
      return result;
    }
  }

  stage = Stage::kInitialized;

  return GXF_SUCCESS;
}

gxf_result_t EntityItem::deinitialize() {
  if (stage != Stage::kDeinitializationInProgress) {
    return GXF_INVALID_LIFECYCLE_STAGE;
  }

  gxf_result_t code = GXF_SUCCESS;
  for (auto riter = components.rbegin(); riter != components.rend(); riter++) {
    auto component = static_cast<Component*>((*riter).value().component_pointer);
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

gxf_result_t EntityItem::destroy(ComponentFactory* factory) {
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

  std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
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
  std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
  // check if gid exists
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("EntityGroup with gid: %05zu is not created yet, "
                  "cannot add entity [eid: %05zu] into non-existent group", gid, eid);
    return GXF_ENTITY_GROUP_NOT_FOUND;
  }
  // check if eid exists
  const auto e_it = entities_.find(eid);
  if (e_it == entities_.end()) {
    GXF_LOG_ERROR("Cannot add non-existent entity [eid: %05zu] into EntityGroup [gid: %05zu]",
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
      GXF_LOG_ERROR("Entity [eid: %05zu] is not initialized to default EntityGroup", eid);
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
    GXF_LOG_ERROR("Cannot remove non-existent entity [eid: %05zu] from its EntityGroup", eid);
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
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existent EntityGroup [gid: %05zu]", eid, gid);
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
  EntityWarden::getEntityGroupResources(gxf_uid_t eid) const {
  FixedVector<gxf_uid_t, kMaxComponents> resource_cids;
  std::shared_lock<std::shared_timed_mutex> lock(shared_mutex_);
  // check if eid exists
  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    GXF_LOG_ERROR("Cannot find entity [eid: %05zu]", eid);
    return Unexpected{GXF_ENTITY_NOT_FOUND};
  }
  // check if gid exists
  gxf_uid_t gid = kNullUid;
  {
    std::shared_lock<std::shared_mutex> item_lock(it->second->entity_item_mutex_);
    gid = it->second->gid;
  }
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existent EntityGroup [gid: %05zu]", eid, gid);
    return Unexpected{GXF_ENTITY_GROUP_NOT_FOUND};
  }

  // collect all resource components from EntityGroup pointed by eid
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
  std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
  // check if eid exists
  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    GXF_LOG_ERROR("Cannot find entity [eid: %05zu]", eid);
    return GXF_ENTITY_NOT_FOUND;
  }
  // check if gid exists
  it->second->entity_item_mutex_.lock_shared();
  gxf_uid_t gid = it->second->gid;
  it->second->entity_item_mutex_.unlock_shared();
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existent EntityGroup [gid: %05zu]", eid, gid);
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
  std::unique_lock<std::shared_timed_mutex> lock(shared_mutex_);
  // check if eid exists
  const auto it = entities_.find(eid);
  if (it == entities_.end()) {
    GXF_LOG_ERROR("Cannot find entity [eid: %05zu]", eid);
    return GXF_ENTITY_NOT_FOUND;
  }
  // check if gid exists
  it->second->entity_item_mutex_.lock_shared();
  gxf_uid_t gid = it->second->gid;
  it->second->entity_item_mutex_.unlock_shared();
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existent EntityGroup [gid: %05zu]", eid, gid);
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

Expected<gxf_uid_t> EntityWarden::entityFindEntityGroupId(gxf_uid_t eid) const {
  // check if eid exists
  const auto e_it = entities_.find(eid);
  if (e_it == entities_.end()) {
    GXF_LOG_ERROR("Non-existent entity [eid: %05zu]", eid);
    return Unexpected{GXF_ENTITY_NOT_FOUND};
  }
  // check if gid exists
  gxf_uid_t gid = e_it->second->gid;
  if (gid == kUnspecifiedUid) {
    GXF_LOG_ERROR("Entity [eid: %05zu] has no EntityGroup", eid);
    return Unexpected{GXF_FAILURE};
  }
  return gid;
}

Expected<const char*> EntityWarden::entityFindEntityGroupName(gxf_uid_t eid) const {
  // check if eid exists
  auto maybe_gid = entityFindEntityGroupId(eid);
  if (!maybe_gid) { return ForwardError(maybe_gid); }

  gxf_uid_t gid = maybe_gid.value();
  const auto eg_it = entity_groups_.find(gid);
  if (eg_it == entity_groups_.end()) {
    GXF_LOG_ERROR("Entity [eid: %05zu] holds non-existent EntityGroup [gid: %05zu]", eid, gid);
    return Unexpected{GXF_ENTITY_GROUP_NOT_FOUND};
  }
  return eg_it->second->name.c_str();
}

gxf_result_t EntityWarden::getEntityRefCount(gxf_uid_t eid, int64_t* count) const {
  if (count == nullptr) return GXF_ARGUMENT_NULL;
  {
    std::shared_lock<std::shared_mutex> ref_count_lock(ref_count_mutex_);
    auto itr = ref_count_store_.find(eid);
    if (itr != ref_count_store_.end()) {
      *count = itr->second.load();
      return GXF_SUCCESS;
    }
  }
  return GXF_PARAMETER_NOT_FOUND;
}

gxf_result_t EntityWarden::decEntityRefCount(gxf_uid_t eid, int64_t& value) {
  std::shared_lock<std::shared_mutex> ref_count_lock(ref_count_mutex_);
  auto itr = ref_count_store_.find(eid);
  if (itr == ref_count_store_.end()) {
    // entity ref count does not exist
    GXF_LOG_ERROR("[E%05" PRId64 "] Ref count for the entity is 0. Cannot decrement", eid);
    return GXF_REF_COUNT_NEGATIVE;
  } else {
    value = (--(itr->second));
  }
  if (value < 0) {
    GXF_LOG_ERROR("[E%05" PRId64 "] Ref count for the entity < 0. Count: %" PRId64 "",
     eid, value);
    return GXF_REF_COUNT_NEGATIVE;
  }
  return GXF_SUCCESS;
}

gxf_result_t EntityWarden::incEntityRefCount(gxf_uid_t eid) {
  {
    std::shared_lock<std::shared_mutex> ref_count_lock(ref_count_mutex_);
    auto itr = ref_count_store_.find(eid);
    if (itr != ref_count_store_.end()) {
      itr->second++;
      return GXF_SUCCESS;
    }
  }
  {
    // Supposed to reach here only once for an entity
    std::unique_lock<std::shared_mutex> ref_count_lock(ref_count_mutex_);
    ref_count_store_.emplace(eid, 1);
  }
  return GXF_SUCCESS;
}

void EntityWarden::removeEntityRefCount(gxf_uid_t eid) {
  std::unique_lock<std::shared_mutex> ref_count_lock(ref_count_mutex_);
  ref_count_store_.erase(eid);
}

}  // namespace gxf
}  // namespace nvidia
