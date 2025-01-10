/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/core/runtime.hpp"

#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvidia {
namespace gxf {

namespace {
constexpr const char* kCorePropertyRefCount = "__ref_count";

// Tid for nvidia::gxf::Component
constexpr gxf_tid_t kComponentTid{0x75bf23d5199843b7, 0xbaaf16853d783bd1};

// dictionary keys in extension metadata file
constexpr const char* kAttributeComponents = "components";
constexpr const char* kAttributeTypeName = "typename";
constexpr const char* kAttributeTypeId = "type_id";
constexpr const char* kAttributeBaseTypeName = "base_typename";

// FIXME prototype code

}  // namespace

gxf_context_t Runtime::context() {
  // return ToContext(this);
  return static_cast<gxf_context_t>(this);
}

gxf_result_t SharedContext::create(gxf_context_t context) {
  parameters_ = std::make_shared<ParameterStorage>(context);
  warden_.setParameterStorage(parameters_);
  warden_.createDefaultEntityGroup(this->getNextId());
  registrar_.setParameterStorage(parameters_);
  registrar_.setParameterRegistrar(&parameter_registrar_);
  resource_registrar_ = std::make_shared<ResourceRegistrar>(context);
  resource_manager_ = std::make_shared<ResourceManager>(context);
  registrar_.setResourceManager(resource_manager_);
  registrar_.setResourceRegistrar(resource_registrar_);

  extension_loader_.initialize(context);

  return GXF_SUCCESS;
}

gxf_result_t SharedContext::initialize(Runtime* rt) {
  gxf_result_t code;
  code = rt->GxfSetExtensionLoader(&extension_loader_);
  if (code != GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetEntityWarden(&warden_);
  if (code != GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetTypeRegistry(&type_registry_);
  if (code != GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetParameterStorage(parameters_);
  if (code != GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetRegistrar(&registrar_);
  if (code != GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetParameterRegistrar(&parameter_registrar_);
  if (code!= GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetResourceRegistrar(resource_registrar_);
  if (code!= GXF_SUCCESS) {
    return code;
  }
  code = rt->GxfSetResourceManager(resource_manager_);
  if (code!= GXF_SUCCESS) {
    return code;
  }
  return code;
}

gxf_result_t SharedContext::destroy() {
  gxf_result_t code;
  code = warden_.cleanup(&extension_loader_);
  if (code != GXF_SUCCESS) {
    return code;
  }
  parameters_.reset();
  return ToResultCode(extension_loader_.unloadAll());
}

gxf_uid_t SharedContext::getNextId() { return next_id_++; }

gxf_result_t SharedContext::removeComponentPointers(
    const FixedVector<gxf_uid_t, kMaxComponents>& cids) {
  // Remove component pointers from the global objects_ map
  std::unique_lock<std::shared_timed_mutex> lock(global_object_mutex_);
  for (auto cid : cids) {
    objects_.erase(cid.value());
  }
  return GXF_SUCCESS;
}

gxf_result_t SharedContext::removeSingleComponentPointer(gxf_uid_t& cid) {
  // Remove component pointer from the global objects_ map
  std::unique_lock<std::shared_timed_mutex> lock(global_object_mutex_);
    objects_.erase(cid);
  return GXF_SUCCESS;
}


gxf_result_t SharedContext::addComponent(gxf_uid_t cid, void* raw_pointer) {
  std::unique_lock<std::shared_timed_mutex> lock(global_object_mutex_);
  objects_[cid] = raw_pointer;

  return GXF_SUCCESS;
}

gxf_result_t SharedContext::findComponentPointer(gxf_context_t context, gxf_uid_t uid,
                                                 void** pointer) {
  *pointer = nullptr;
  std::unordered_map<gxf_uid_t, void*>::iterator it;
  {
    std::shared_lock<std::shared_timed_mutex> lock(global_object_mutex_);
    it = objects_.find(uid);
  }

  if (it != objects_.end()) {
    *pointer = it->second;
    return GXF_SUCCESS;
  }

  // Component not found in shared context, search for it using entity item
  gxf_uid_t eid = kNullUid;
  auto result = GxfComponentEntity(context, uid, &eid);
  if (result != GXF_SUCCESS) { return result; }

  EntityItem* item_ptr = nullptr;
  result = GxfEntityGetItemPtr(context, eid, reinterpret_cast<void**>(&item_ptr));
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find Entity Item for Entity %lu, component %lu", eid, uid);
    return result;
  }
  std::shared_lock<std::shared_mutex> item_lock(item_ptr->entity_item_mutex_);
  bool found_comp_ptr = false;
  for (auto comp : item_ptr->components) {
    if (uid == comp.value().cid) {
      *pointer = comp.value().raw_pointer;
      found_comp_ptr = true;
      break;
    }
  }
  if (found_comp_ptr == false) {
    GXF_LOG_ERROR("Could not find component pointer for Entity %lu, component %lu", eid, uid);
    return GXF_ENTITY_COMPONENT_NOT_FOUND;
  }
  // This line is needed ideally, but giving perf degradation in some cases
  // Uncomment after thorough analysis
  // addComponent(uid, *pointer);
  return GXF_SUCCESS;
}

gxf_result_t SharedContext::loadExtensionImpl(const std::string& filename) {
  std::unique_lock<std::mutex> lock(load_extension_mutex_);
  const auto result = extension_loader_.load(filename.c_str());
  if (!result) {
    return result.error();
  }

  return GXF_SUCCESS;
}

gxf_result_t SharedContext::loadExtensionImpl(Extension& extension) {
  std::unique_lock<std::mutex> lock(load_extension_mutex_);

  const auto result = extension_loader_.load(&extension);
  if (!result) {
    return result.error();
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::create() {
  shared_context_ = new SharedContext();

  shared_context_owner_ = true;
  shared_context_->create(context());
  shared_context_->initialize(this);

  program_.setup(context(), warden_, &entity_executor_, parameters_);

  gxf_result_t code;

  code = GxfRegisterComponent(kComponentTid, "nvidia::gxf::Component", "");
  if (code != GXF_SUCCESS) return code;

  code = GxfComponentTypeId(TypenameAsString<Component>(), &component_tid_);
  if (code != GXF_SUCCESS) return code;

  return GXF_SUCCESS;
}

gxf_result_t Runtime::create(gxf_context_t shared) {
  shared_context_ = static_cast<SharedContext*>(shared);

  shared_context_owner_ = false;
  shared_context_->initialize(this);

  program_.setup(context(), warden_, &entity_executor_, parameters_);

  gxf_result_t code;

  code = GxfComponentTypeId(TypenameAsString<Component>(), &component_tid_);
  if (code != GXF_SUCCESS) return code;

  return GXF_SUCCESS;
}

gxf_result_t Runtime::destroy() {
  gxf_result_t code = GXF_SUCCESS;

  program_.destroy();  // FIXME handle error code

  // FIXME deinit happens in block here because entities might have
  // dependencies on each other
  if (shared_context_owner_) {
    code = shared_context_->destroy();
    if (code != GXF_SUCCESS) {
      return code;
    }
    delete shared_context_;
    shared_context_ = nullptr;
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfGetSharedContext(gxf_context_t* shared) {
  *shared = static_cast<gxf_context_t>(shared_context_);
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfRegisterComponent(gxf_tid_t tid, const char* name, const char* base) {
  Expected<void> result;

  result = type_registry_->add(tid, name);
  if (!result) {
    GXF_LOG_VERBOSE("Could not register component '%s'. Did you register it twice?", name);
    return result.error();
  }

  if (base == nullptr || std::strcmp(base, "") == 0) {
    parameter_registrar_->addParameterlessType(tid, std::string(name));
    return GXF_SUCCESS;
  }

  result = type_registry_->add_base(name, base);
  if (!result) {
    GXF_LOG_VERBOSE("Base class '%s' was not registered. Did you forget to register it?", base);
    return result.error();
  }

  // Non nvidia::gxf::Component elements do not have any parameters
  auto base_result = type_registry_->is_base(tid, kComponentTid);
  if (!base_result) { return base_result.error(); }
  if (!base_result.value()) {
    parameter_registrar_->addParameterlessType(tid, std::string(name));
    return GXF_SUCCESS;
  }

  Expected<void*> raw_pointer = extension_loader_->allocate(tid);

  // Abstract nvidia::gxf::Components do not have any parameters either
  if (!raw_pointer && raw_pointer.error() == GXF_FACTORY_ABSTRACT_CLASS) {
    parameter_registrar_->addParameterlessType(tid, std::string(name));
    return GXF_SUCCESS;
  } else if (!raw_pointer) {
    GXF_LOG_ERROR("Failed to create component %s", name);
    return GXF_FAILURE;
  }

  // use Registrar public member to pass arguments for each component
  // for Registrar.parameter() and Registrar.resource() in Component::registerInterface()
  registrar_->tid = tid;
  registrar_->type_name.assign(name);
  // Parameters: register component type and its member Parameter<T> types
  // set Registrar.ParameterRegistrar != nullptr & a mock Registrar.ParameterStorage
  // This is not the stage to populate ParameterStorage, so
  // Use a temporary parameter storage for component registration
  std::shared_ptr<ParameterStorage> tmp_param_storage =
    std::make_shared<ParameterStorage>(context());
  registrar_->setParameterStorage(tmp_param_storage);

  Component* component = reinterpret_cast<Component*>(raw_pointer.value());
  component->internalSetup(nullptr, 0, 1, nullptr);
  // Resource: register component type and its member Resource<Handle<T>> types
  // set Registrar.ResourceRegistrar != nullptr && Registrar.ResourceManager == nullptr
  // ResourceManager is not ready till now, will connect it after register stage
  registrar_->setResourceManager(nullptr);
  // First time Runtime call Component::registerInterface(). Twice total.
  const auto registration_result = component->registerInterface(registrar_);
  result = extension_loader_->deallocate(tid, component);

  // Reset parameter storage used by runtime
  registrar_->setParameterStorage(parameters_);
  registrar_->setResourceManager(resource_manager_);

  if (registration_result != GXF_SUCCESS) {
    GXF_LOG_VERBOSE("Failed to register interface for component: %s", name);
    return registration_result;
  }

  if (!result) {
    GXF_LOG_VERBOSE("Failed to deallocate component: %s", name);
    return result.error();
  }

  GXF_LOG_VERBOSE("Successfully registered component [%s] with base type [%s]", name, base);
  return GXF_SUCCESS;
}


gxf_result_t Runtime::GxfRegisterComponentInExtension(const gxf_tid_t& component_tid,
                                                      const gxf_tid_t& extension_tid) {
  Expected<void> result = extension_loader_->registerRuntimeComponent(component_tid, extension_tid);
  if (!result) { return ToResultCode(result); }

  gxf_component_info_t info;
  result = extension_loader_->getComponentInfo(component_tid, &info);
  if (!result) { return ToResultCode(result); }

  return GxfRegisterComponent(component_tid, info.type_name, info.base_name);
}

// Gets version information about Runtime and list of loaded Extensions.
gxf_result_t Runtime::GxfRuntimeInfo(gxf_runtime_info* info) {
  if (!info) {
    GXF_LOG_ERROR("Received null pointer for Runtime Info query");
    return GXF_NULL_POINTER;
  }
  info->version = gxf_core_version_.c_str();
  return ToResultCode(extension_loader_->getExtensions(&(info->num_extensions), info->extensions));
}

// Gets description of loaded extension and list of components it provides
gxf_result_t Runtime::GxfExtensionInfo(const gxf_tid_t eid, gxf_extension_info_t* info) {
  if (!info) {
    GXF_LOG_VERBOSE("Invalid Parameter");
    return GXF_NULL_POINTER;
  }
  return ToResultCode(extension_loader_->getExtensionInfo(eid, info));
}

// Gets description of component and list of parameter. List parameter is only
// available if the component is already instantiated.
gxf_result_t Runtime::GxfComponentInfo(const gxf_tid_t tid, gxf_component_info_t* info) {
  if (!info) {
    GXF_LOG_VERBOSE("Received null pointer for Component Info query");
    return GXF_NULL_POINTER;
  }
  auto result = extension_loader_->getComponentInfo(tid, info);
  if (!result) {
    return ToResultCode(result);
  }

  auto base_result = type_registry_->is_base(tid, kComponentTid);
  if (!base_result) { return base_result.error(); }
  const bool is_component = base_result.value();
  // Fills Parameter Information
  if (!result || info->is_abstract || !is_component) {
    // Abstract components and pure base type components(Ex: Tensor) don't have parameters
    info->num_parameters = 0;
    info->parameters = nullptr;
    parameter_registrar_->addParameterlessType(tid, std::string(info->type_name));
    return ToResultCode(result);
  }

  if (!parameter_registrar_->hasComponent(tid)) {
    return GXF_ENTITY_COMPONENT_NOT_FOUND;
  }

  size_t count = parameter_registrar_->componentParameterCount(tid);
  if (count > info->num_parameters) {
    // Not enough space to output parameter info, return the required memory size
    info->num_parameters = count;
    return GXF_QUERY_NOT_ENOUGH_CAPACITY;
  }

  // Fills Parameter Info
  info->num_parameters = count;
  result = parameter_registrar_->getParameterKeys(tid, info->parameters, count);
  if (!result) {
    return ToResultCode(result);
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfParameterInfo(const gxf_tid_t cid, const char* key,
                      gxf_parameter_info_t* info) {
  if (!info) {
    GXF_LOG_VERBOSE("Invalid Parameter");
    return GXF_ARGUMENT_NULL;
  }

  auto result = extension_loader_->getParameterInfo(cid, key, info);

  if (!result) {
    return ToResultCode(result);
  }
  return GXF_SUCCESS;
}
gxf_result_t Runtime::GxfGetParameterInfo(const gxf_tid_t cid, const char* key,
                      gxf_parameter_info_t* info) {
  if (!info) {
    GXF_LOG_ERROR("Received null pointer for Parameter info query");
    return GXF_NULL_POINTER;
  }
  if (!parameter_registrar_->hasComponent(cid)) {
    // Tries to load component info including parameter info
    gxf_component_info_t component_info;
    component_info.num_parameters = 0;
    auto result = GxfComponentInfo(cid, &component_info);
    if (result != GXF_QUERY_NOT_ENOUGH_CAPACITY) {
      return result;
    }
  }

  if (!parameter_registrar_->hasComponent(cid)) {
    GXF_LOG_ERROR("Parameter %s not found in component (type=%016lx%016lx)",
                  key, cid.hash1, cid.hash2);
    return GXF_PARAMETER_NOT_FOUND;
  }

  // Fills Parameter info
  auto result = parameter_registrar_->getParameterInfo(cid, key, info);
  if (!result) {
    return ToResultCode(result);
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::loadExtensionImpl(const std::string& filename) {
  gxf_result_t code = shared_context_->loadExtensionImpl(filename);
  if (code != GXF_SUCCESS) {
    GXF_LOG_VERBOSE("Error: Could not load extension '%s'", filename.c_str());
    return code;
  }
  GXF_LOG_VERBOSE("Loaded extension '%s'", filename.c_str());
  return code;
}

gxf_result_t Runtime::loadExtensionImpl(Extension& extension) {
  gxf_result_t code = shared_context_->loadExtensionImpl(extension);
  if (code != GXF_SUCCESS) {
    GXF_LOG_VERBOSE("Error: Could not load extension");
    return code;
  }
  return code;
}

gxf_result_t Runtime::SearchLdLibraryPath(const std::string& filename) {
  // find standard paths to search for libraries
  std::vector<std::string> standard_paths;
  char path_delimiter = ':';
  auto tokenize = [&standard_paths, &path_delimiter](const std::string& string_path) {
    size_t start;
    size_t end = 0;

    while ((start = string_path.find_first_not_of(path_delimiter, end)) != std::string::npos) {
        end = string_path.find(path_delimiter, start);
        standard_paths.push_back(string_path.substr(start, end - start));
    }
  };

  auto env_pointer = std::getenv("LD_LIBRARY_PATH");
  if (env_pointer != nullptr) {
    auto ld_path = std::string(env_pointer);
    GXF_LOG_DEBUG("LD_LIBRARY_PATH found in the env: %s", ld_path.c_str());
    if (ld_path.size() != 0) {
      tokenize(ld_path);
    }
  }

  auto base_filename = filename.substr(filename.find_last_of('/') + 1);
  gxf_result_t error_code = GXF_EXTENSION_FILE_NOT_FOUND;
  // search for libraries in ld library paths
  for (const auto& prefix : standard_paths) {
    std::string filepath{prefix + "/" + base_filename};
    if ((access(filepath.c_str(), X_OK) == 0) && (access(filepath.c_str(), R_OK) == 0)) {
      GXF_LOG_INFO("Trying extension %s found in ld library path %s", base_filename.c_str(),
                    prefix.c_str());
      error_code = loadExtensionImpl(filepath);
      if (error_code == GXF_SUCCESS) {
        GXF_LOG_INFO("Loaded extension %s from ld library path '%s'", base_filename.c_str(),
                      prefix.c_str());
        break;
      }
    }
  }

  return error_code;
}

gxf_result_t Runtime::GxfLoadExtensions(const GxfLoadExtensionsInfo& info) {
  // Make sure that the prefix also has a / to be on the safe side.
  std::string prefix;
  if (info.base_directory != nullptr) {
    prefix = info.base_directory;
    if (!prefix.empty()) {
      prefix += "/";
    }
  }

  // Load individual extensions first
  if (info.extension_filenames_count > 0 && info.extension_filenames == nullptr) {
    GXF_LOG_VERBOSE("Error: extension_filenames is null");
    return GXF_ARGUMENT_NULL;
  }
  for (uint32_t i = 0; i < info.extension_filenames_count; i++) {
    auto result = loadExtensionImpl(prefix + info.extension_filenames[i]);

    // Try searching in LD_LIBRARY_PATH if extension is not found
    if (result == GXF_EXTENSION_FILE_NOT_FOUND) {
      result = SearchLdLibraryPath(info.extension_filenames[i]);
    }

    if (result != GXF_SUCCESS) {
      return result;
    }
  }

  // Load extensions from manifests
  if (info.manifest_filenames_count > 0 && info.manifest_filenames == nullptr) {
    GXF_LOG_VERBOSE("Error: manifest_filenames is null");
    return GXF_ARGUMENT_NULL;
  }
  for (uint32_t i = 0; i < info.manifest_filenames_count; i++) {
    const std::string manifest_filename = prefix + info.manifest_filenames[i];
    try {
      const auto node = YAML::LoadFile(manifest_filename);
      for (const auto& entry : node["extensions"]) {
        auto filename = entry.as<std::string>();
        auto result = loadExtensionImpl(prefix + filename);
        if (result == GXF_EXTENSION_FILE_NOT_FOUND) {
          result = SearchLdLibraryPath(filename);
        }
        if (result != GXF_SUCCESS) {
          return result;
        }
      }
    } catch (std::exception& x) {
      GXF_LOG_VERBOSE("Error loading manifest '%s': %s", manifest_filename.c_str(), x.what());
      return GXF_FAILURE;
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfLoadExtensionFromPointer(Extension* extension) {
  if (extension == nullptr) {
    GXF_LOG_VERBOSE("Error: extension is null");
    return GXF_ARGUMENT_NULL;
  }
  auto result = loadExtensionImpl(*extension);

  return result;
}

gxf_result_t Runtime::GxfGraphLoadFileInternal(const char* filename, const char* entity_prefix,
                                               const char* params_override[],
                                               const uint32_t num_overrides, gxf_uid_t parent_eid,
                                               const YAML::Node& prerequisites) {
  YamlFileLoader loader;
  loader.setParameterStorage(parameters_);
  loader.setFileRoot(this->graph_path_);
  auto result = loader.loadFromFile(context(), filename, entity_prefix, params_override,
                                    num_overrides, parent_eid, prerequisites);
  if (!result) { return ToResultCode(result); }
  GXF_LOG_VERBOSE("Loaded graph file '%s'", filename);
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfGraphLoadFile(const char* filename, const char* params_override[],
                                       const uint32_t num_overrides) {
  return GxfGraphLoadFileInternal(filename, "", params_override, num_overrides);
}

gxf_result_t Runtime::GxfGraphLoadFileExtended(const char* filename, const char* entity_prefix,
                                               const char* params_override[],
                                               const uint32_t num_overrides,
                                               gxf_uid_t parent_eid,
                                               void* prerequisites) {
  YAML::Node* node = static_cast<YAML::Node*>(prerequisites);
  return GxfGraphLoadFileInternal(filename, entity_prefix, params_override, num_overrides,
                                  parent_eid, *node);
}

gxf_result_t Runtime::GxfGraphSetRootPath(const char* path) {
  if (path != nullptr) {
    this->graph_path_ = path;
    return GXF_SUCCESS;
  }
  return GXF_ARGUMENT_NULL;
}

gxf_result_t Runtime::GxfGraphParseString(const char* text,
                     const char* params_override[] , const uint32_t num_overrides) {
  YamlFileLoader loader;
  loader.setParameterStorage(parameters_);
  return ToResultCode(loader.loadFromString(context(), text, "", params_override, num_overrides));
}

gxf_result_t Runtime::GxfGraphSaveToFile(const char* filename) {
  if (filename == nullptr) {
    GXF_LOG_ERROR("File name was null when exporting graph");
    return GXF_ARGUMENT_NULL;
  }
  YamlFileLoader loader;
  loader.setParameterStorage(parameters_);
  auto result = loader.saveToFile(context(), filename);
  if (!result) { return ToResultCode(result); }
  GXF_LOG_INFO("Saved graph to file '%s'", filename);
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfCreateEntity(const GxfEntityCreateInfo& info, gxf_uid_t& eid,
                                      void** item_ptr) {
  if (info.entity_name) {
    // Entity names should be unique.
    // Check whether there is already an entity existing with the same name
    gxf_uid_t eidExisting = kNullUid;
    const gxf_result_t result = GxfEntityFind(info.entity_name, &eidExisting);
    if (result == GXF_SUCCESS) {
      GXF_LOG_ERROR("There is already an entity with the name '%s' eid [E%05" PRId64 "]",
                    info.entity_name, eidExisting);
      return GXF_ARGUMENT_INVALID;
    }
  }

  // Get a new unique entity ID
  eid = shared_context_->getNextId();

  // According to definition of struct GxfEntityCreateInfo
  // names must not start with a double underscore.
  if (info.entity_name != nullptr && info.entity_name[0] == '_' && info.entity_name[1] == '_') {
    GXF_LOG_ERROR("Invalid Entity name: Entity name cannot start with double underscore: %s",
                  info.entity_name);
    return GXF_ARGUMENT_INVALID;
  }

  // Create an entity name if none was given
  std::string entity_name = (info.entity_name && info.entity_name[0] != '\0')
                                      ? info.entity_name
                                      : ("__entity_" + std::to_string(eid));

  GXF_LOG_VERBOSE("[E%05" PRId64 "] CREATE ENTITY '%s'", eid, entity_name.c_str());
  const gxf_result_t result_1 = warden_->create(eid, reinterpret_cast<EntityItem**>(item_ptr),
                                                entity_name);
  if (result_1 != GXF_SUCCESS) {
    return result_1;
  }

  // Add the entity to the program if desired
  if ((info.flags & GXF_ENTITY_CREATE_PROGRAM_BIT) != 0) {
    void* item = item_ptr == nullptr ? nullptr : *(item_ptr);
    const auto result_3 = program_.addEntity(eid, static_cast<EntityItem*>((item)));
    if (!result_3) {
      return ToResultCode(result_3);
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfCreateEntityGroup(const char* name, gxf_uid_t* gid) {
  *gid = shared_context_->getNextId();
  const gxf_result_t result = warden_->createEntityGroup(*gid, name);
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create EntityGroup [gid: %05" PRId64 ", name: %s]", *gid, name);
    return result;
  }
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfUpdateEntityGroup(gxf_uid_t gid, gxf_uid_t eid) {
  gxf_result_t result = warden_->updateEntityGroup(gid, eid);
  if (result != GXF_SUCCESS) {
    return result;
  }
  const char* entity_name = "UNKNOWN";
  GxfEntityGetName(eid, &entity_name);
  GXF_LOG_DEBUG(
      "Entity [eid: %05" PRId64 ", name: %s] updated its EntityGroup to [gid: %05" PRId64 "]",
      eid, entity_name, gid);
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityIsValid(gxf_uid_t eid, bool* valid) {
  *valid = false;
  gxf_result_t code = warden_->isValid(eid);
  if (code == GXF_SUCCESS) {
    *valid = true;
  }
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityActivate(gxf_uid_t eid) {
  GXF_LOG_VERBOSE("[E%05" PRId64 "] ENTITY ACTIVATE ", eid);

  auto entity = nvidia::gxf::Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }

  GXF_LOG_VERBOSE("[E%05" PRId64 "] WARDEN INITIALIZE", eid);
  const gxf_result_t code1 = warden_->initialize(eid);
  if (code1 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not initialize entity '%s' (E%" PRId64 "): %s",
                  entity->name(), eid, GxfResultStr(code1));
    return code1;
  }

  GXF_LOG_VERBOSE("[E%05" PRId64 "] ENTITY EXECUTOR ACTIVATE", eid);
  const gxf_result_t code2 = entity_executor_.activate(context(), eid);
  if (code2 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not activate entity '%s' (E%" PRId64 "): %s",
                  entity->name(), eid, GxfResultStr(code2));
    return code2;
  }

  GXF_LOG_VERBOSE("[E%05" PRId64 "] SCHEDULE ENTITY '%s' ", eid, entity->name());
  const auto code3 = program_.scheduleEntity(eid);
  if (!code3) {
    GXF_LOG_ERROR("Could not schedule entity '%s' (E%" PRId64 ") for execution: %s",
                  entity->name(), eid, GxfResultStr(code3.error()));
    return ToResultCode(code3);
  }
  GXF_LOG_VERBOSE("[E%05" PRId64 "] ENTITY ACTIVATED '%s' ", eid, entity->name());

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityDeactivate(gxf_uid_t eid) {
  GXF_LOG_VERBOSE("[E%05" PRId64 "] ENTITY DEACTIVATE", eid);

  auto entity = nvidia::gxf::Entity::Shared(context(), eid);
  if (!entity) {
    return ToResultCode(entity);
  }

  const auto code1 = program_.unscheduleEntity(eid);
  if (!code1) {
    GXF_LOG_ERROR("Could not unschedule entity '%s' (E%" PRId64 ") from execution: %s",
                  entity->name(), eid, GxfResultStr(code1.error()));
    return ToResultCode(code1);
  }

  const gxf_result_t code2 = entity_executor_.deactivate(eid);
  if (code2 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not deactivate entity '%s' (E%" PRId64 "): %s",
                  entity->name(), eid, GxfResultStr(code2));
    return code2;
  }

  const gxf_result_t code3 = warden_->deinitialize(eid);
  if (code3 != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not deinitialize entity '%s' (E%" PRId64 "): %s",
                  entity->name(), eid, GxfResultStr(code3));
    return code3;
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityDestroyImpl(gxf_uid_t eid) {
  // DO NOT CREATE AN ENTITY OBJECT IN THIS METHOD.
  // This method gets called because the refcount of the
  // entity `eid` dropped to zero.
  // Creating an object at this point will cause the ref
  // count to increase to 1 then again to 0 on exit causing
  // a call to this method again leading to an infinite loop.

  const char* entity_name = nullptr;
  gxf_result_t code = GxfEntityGetName(eid, &entity_name);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to obtain name of entity (E%" PRId64 "): %s", eid, GxfResultStr(code));
  }

  GXF_LOG_VERBOSE("[E%05" PRId64 "] ENTITY DESTROY '%s'", eid, entity_name);

  // Collects component ids for clean up later
  auto cids = warden_->getEntityComponents(eid);
  if (!cids) {
    GXF_LOG_ERROR("Failed to get components for entity '%s' (E%" PRId64 ")  while deleting it: %s",
                  entity_name, eid, GxfResultStr(code));
    return ToResultCode(cids);
  }

  // Deinitialize first
  code = warden_->deinitialize(eid);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to deinitialize entity '%s' (E%" PRId64 "): %s", entity_name, eid,
                  GxfResultStr(code));
    return code;
  }

  code = shared_context_->removeComponentPointers(cids.value());
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to destroy entity '%s' (E%" PRId64 "): %s", entity_name, eid,
                  GxfResultStr(code));
    return code;
  }

  // Destroy
  code = warden_->destroy(eid, extension_loader_);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to destroy entity '%s' (E%" PRId64 "): %s", entity_name, eid,
                  GxfResultStr(code));
    return code;
  }

  // Clear parameters for components and entities
  Expected<void> result = Success;
  for (auto cid : cids.value()) {
    result = parameters_->clearEntityParameters(cid.value());
    const char* component_name;
    code = GxfComponentName(cid.value(), &component_name);
    if (code != GXF_SUCCESS) {
      component_name = "";
    }
    if (!result) {
      GXF_LOG_ERROR("Failed to clear parameters for component '%s/%s' (C%" PRId64 "): %s",
                    entity_name, component_name, cid.value(), GxfResultStr(result.error()));
      return ToResultCode(result);
    }
  }
  result = parameters_->clearEntityParameters(eid);
  if (!result) {
    GXF_LOG_ERROR("Failed to clear parameters for entity '%s' (E%" PRId64 "): %s", entity_name, eid,
                  GxfResultStr(result.error()));
  }
  warden_->removeEntityRefCount(eid);
  return ToResultCode(result);
}

gxf_result_t Runtime::GxfEntityDestroy(gxf_uid_t eid) {
  // Check reference count
  int64_t count = 0;  // FIXME
  const auto code = GxfEntityGetRefCount(eid, &count);
  if (code == GXF_PARAMETER_NOT_FOUND) {
    count = 0;
  } else if (code != GXF_SUCCESS) {
    return code;
  }
  if (count != 0) {
    return GXF_FAILURE;
  } else {
    return GxfEntityDestroyImpl(eid);
  }
}

gxf_result_t Runtime::GxfEntityFind(const char* name, gxf_uid_t* eid) {
  return warden_->find(context(), name, eid);
}

gxf_result_t Runtime::GxfEntityFindAll(uint64_t* num_entities, gxf_uid_t* entities) {
  const uint64_t max_entities = (*num_entities);

  const auto maybe_entities_vector = warden_->getAll();
  if (!maybe_entities_vector) {
    GXF_LOG_ERROR("Failed to retrieve entities vector from EntityWarden");
    return GXF_FAILURE;
  }

  const FixedVector<gxf_uid_t, kMaxEntities>& entities_vector = maybe_entities_vector.value();
  // Set the output variable to the number of entities in the application
  (*num_entities) = entities_vector.size();

  if (max_entities < entities_vector.size()) {
    GXF_LOG_ERROR("Entities buffer capacity %" PRIu64 ", but application contains %zu entities",
                  max_entities, entities_vector.size());
    return GXF_QUERY_NOT_ENOUGH_CAPACITY;
  }

  std::copy(entities_vector.data(), entities_vector.data() +
            entities_vector.size(), entities);

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityRefCountInc(gxf_uid_t eid) {
  return warden_->incEntityRefCount(eid);
}

gxf_result_t Runtime::GxfEntityRefCountDec(gxf_uid_t eid) {
  int64_t value = 0;
  const auto code = warden_->decEntityRefCount(eid, value);
  if (code != GXF_SUCCESS) { return code; }
  if (value == 0) {
    return GxfEntityDestroyImpl(eid);
  } else {
    return GXF_SUCCESS;
  }
}

gxf_result_t Runtime::GxfEntityGetRefCount(gxf_uid_t eid, int64_t* count) const {
  if (count == nullptr) return GXF_ARGUMENT_NULL;
  return warden_->getEntityRefCount(eid, count);
}

gxf_result_t Runtime::GxfEntityGetStatus(gxf_uid_t eid, gxf_entity_status_t* entity_status) {
  auto result = entity_executor_.getEntityStatus(eid, entity_status);
  if (!result) {
    GXF_LOG_VERBOSE("[E%05" PRId64 "] Entity status query failed with error %s", eid,
                    GxfResultStr(ToResultCode(result)));
  }
  return ToResultCode(result);
}

gxf_result_t Runtime::GxfEntityGetName(gxf_uid_t eid, const char** entity_name) {
  auto result = warden_->getEntityName(eid, entity_name);
  if (!result) {
    GXF_LOG_VERBOSE("[E%05" PRId64 "] Entity name query failed with error %s", eid,
                    GxfResultStr(result));
  }
  return result;
}

gxf_result_t Runtime::GxfEntityGetState(gxf_uid_t eid, entity_state_t* b_status) {
  entity_state_t entity_behavior_status;
  const gxf_result_t code1 = entity_executor_.getEntityBehaviorStatus(eid, entity_behavior_status);
  if (code1 != GXF_SUCCESS) {
    GXF_LOG_VERBOSE("[E%05" PRId64 "] Cannot query the node's behavior status", eid);
    return code1;
  }
  *b_status = entity_behavior_status;
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityNotifyEventType(gxf_uid_t eid, gxf_event_t event) {
  return ToResultCode(program_.entityEventNotify(eid, event));
}

gxf_result_t Runtime::GxfComponentTypeName(gxf_tid_t tid, const char** name) {
  if (name == nullptr) {return GXF_NULL_POINTER; }
  const Expected<const char*> result = type_registry_->name(tid);
  if (result) {
    *name = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfComponentTypeNameFromUID(gxf_uid_t cid, const char** name) {
  gxf_tid_t tid = GxfTidNull();
  auto code = GxfComponentType(cid, &tid);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component type for component [C%05" PRId64 "]", cid);
    return code;
  }

  code = GxfComponentTypeName(tid, name);
  if (code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Could not find component type name for component [C%05" PRId64 "]", cid);
  }

  return code;
}

gxf_result_t Runtime::GxfComponentTypeId(const char* name, gxf_tid_t* tid) {
  const Expected<gxf_tid_t> result = type_registry_->id_from_name(name);
  if (result) {
    *tid = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfComponentName(gxf_uid_t cid, const char** name) {
  return GxfParameterGetStr(cid, kInternalNameParameterKey, name);
}

gxf_result_t Runtime::GxfComponentEntity(gxf_uid_t cid, gxf_uid_t* eid) {
  const Expected<gxf_uid_t> result = warden_->getComponentEntity(cid);
  if (result) {
    *eid = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfEntityGetItemPtr(gxf_uid_t eid, void** ptr) {
  const Expected<EntityItem*> result = warden_->getEntityPtr(eid);
  if (result) {
    *ptr = reinterpret_cast<void*>(result.value());
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfComponentIsBase(gxf_tid_t derived, gxf_tid_t base, bool* result) {
  auto code = type_registry_->is_base(derived, base);
  if (!code) { return code.error(); }

  *result = code.value();
  return GXF_SUCCESS;
}


gxf_result_t Runtime::GxfComponentAdd(gxf_uid_t eid, gxf_tid_t tid, const char* name,
                                      gxf_uid_t* out_cid, void** comp_ptr) {
  // FIXME(dweikersdorfer) Add this check back.
  // // Special case: System
  // if (tid == sys_tid_ && state_ == State::RUNNING) {
  //   GXF_LOG_VERBOSE("ERROR: Can not create System component after GxfRun");
  //   return GXF_FAILURE;
  // }

  gxf_tid_t codelet_tid;
  auto static_code = GxfComponentTypeId(TypenameAsString<Codelet>(), &codelet_tid);
  if (static_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Standard extension has not been loaded!");
    return static_code;
  }

  // Verify entity
  auto code = warden_->isValid(eid);
  if (code != GXF_SUCCESS) { return code; }

  // Allocate component
  Expected<void*> raw_pointer = extension_loader_->allocate(tid);
  if (!raw_pointer) {
    return raw_pointer.error();
  }

  const gxf_uid_t cid = shared_context_->getNextId();
  Component* component = nullptr;
  Expected<bool> base_result = false;

  auto type_name = type_registry_->name(tid);
  if (!type_name) { return ToResultCode(type_name); }
  GXF_LOG_VERBOSE("[E%05" PRId64 "] COMPONENT CREATE: C%05" PRId64 " (type=%s) name: %s",
                  eid, cid, type_name.value(), name);

  base_result = type_registry_->is_base(tid, component_tid_);

  if (!base_result) { return base_result.error(); }
  if (base_result.value()) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    component = reinterpret_cast<Component*>(raw_pointer.value());
    component->internalSetup(context(), eid, cid, registrar_);
    // use Registrar public member to pass arguments for each component
    // for Registrar.parameter() and Registrar.resource() in Component::registerInterface()
    registrar_->tid = tid;
    registrar_->cid = cid;
    registrar_->setParameterRegistrar(nullptr);
    // set Registrar.ResourceRegistrar = nullptr && Registrar.ResourceManager != nullptr
    // Resources have already been registered when extension was loaded
    registrar_->setResourceRegistrar(nullptr);
    // Second time Runtime call Component::registerInterface(). Twice total.
    const auto registration_result = component->registerInterface(registrar_);
    if (registration_result != GXF_SUCCESS) { return registration_result; }
    registrar_->setParameterRegistrar(parameter_registrar_);
    registrar_->setResourceRegistrar(resource_registrar_);
  }

  if (name) {
    if (strlen(name) >= kMaxComponentNameSize) {
      GXF_LOG_ERROR("Component name exceeds max limit of %d characters", kMaxComponentNameSize);
      return GXF_ENTITY_COMPONENT_NAME_EXCEEDS_LIMIT;
    }
    GxfParameterSetStr(cid, kInternalNameParameterKey, name);
  } else {
    GxfParameterSetStr(cid, kInternalNameParameterKey, "");
  }

  code = warden_->addComponent(eid, cid, tid, raw_pointer.value(), component);
  if (code != GXF_SUCCESS) {
    return code;
  }

  // code = shared_context_->addComponent(cid, raw_pointer.value());
  // if (code != GXF_SUCCESS) {
  //   return code;
  // }

  *out_cid = cid;
  *comp_ptr = (raw_pointer.value());
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfComponentAddWithItem(void* item_ptr, gxf_tid_t tid, const char* name,
                                               gxf_uid_t* out_cid, void** comp_ptr) {
  EntityItem* item = static_cast<EntityItem*>(item_ptr);
  gxf_uid_t eid = item->uid;
  return GxfComponentAdd(eid, tid, name, out_cid, comp_ptr);
}

gxf_result_t Runtime::GxfComponentRemove(gxf_uid_t eid, gxf_tid_t tid, const char * name) {
  gxf_uid_t cid = kNullUid;
  Expected<EntityItem*> entity_item = warden_->getEntityPtr(eid);
  if (!entity_item) { return ToResultCode(entity_item); }
  void* ptr;
  auto result = warden_->findComponent(context(), entity_item.value(), tid, name,
                                       nullptr, type_registry_, &cid, &ptr);
  if (result != GXF_SUCCESS || cid == kNullUid) {
    const char* entity_name = "UNKNOWN";
    GxfEntityGetName(eid, &entity_name);
    const char* component_type = "UNKNOWN";
    GxfComponentTypeName(tid, &component_type);
    GXF_LOG_ERROR("Failed to find component with name %s , type id %s from entity %s.",
    name, component_type, entity_name);
    return result;
  }
  return GxfComponentRemove(cid);
}

gxf_result_t Runtime::GxfComponentRemove(gxf_uid_t cid) {
  gxf_tid_t codelet_tid;
  auto static_code = GxfComponentTypeId(TypenameAsString<Codelet>(), &codelet_tid);
  if (static_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Standard extension has not been loaded!");
    return static_code;
  }

  if (cid == kNullUid || cid == kUnspecifiedUid) {
    GXF_LOG_ERROR("Component id not provided for component removal, returning.");
    return GXF_ARGUMENT_INVALID;
  }
  // Obtain the entity id
  auto eid = warden_->getComponentEntity(cid);
  if (!eid) {
    const char * component_name;
    auto name_result = parameters_->getStr(cid, kInternalNameParameterKey);
    if (name_result) {
      component_name = name_result.value();
      GXF_LOG_ERROR("Could not find the entity for component %s.", component_name);
    } else {
      GXF_LOG_ERROR("Coult not find the entity for component id %lu.", cid);
    }
    return eid.error();
  }
  // Remove component from entity warden
  gxf_result_t result = GXF_FAILURE;
  result = warden_->removeComponent(context(), eid.value(), cid, extension_loader_);
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Error while removing component id %lu.", cid);
    return result;
  }
  // Remove component from global object storage
  result = shared_context_->removeSingleComponentPointer(cid);
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to remove component %s", GxfResultStr(result));
    return result;
  }

  // Remove parameters stored for this component
  auto code = parameters_->clearEntityParameters(cid);
  if (!code) {
    auto name_result = parameters_->getStr(cid, kInternalNameParameterKey);
    if (name_result) {
      GXF_LOG_ERROR("Could not find the entity for component %s.", name_result.value());
    } else {
      GXF_LOG_ERROR("Could not find the entity for component id %lu.", cid);
    }
    return code.error();
  }

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfComponentAddToInterface(gxf_uid_t eid, gxf_uid_t cid,
                                                 const char* name) {
  // Find entity
  gxf_result_t code = warden_->isValid(eid);
  if (code != GXF_SUCCESS) {
    return code;
  }
  return warden_->addComponentToInterface(eid, cid, name);
}

gxf_result_t Runtime::GxfComponentFind(gxf_uid_t eid, gxf_tid_t tid, const char* name,
                                       int32_t* offset, gxf_uid_t* cid) {
  Expected<EntityItem*> entity_item = warden_->getEntityPtr(eid);
  if (!entity_item) { return ToResultCode(entity_item); }
  void* ptr;
  return warden_->findComponent(context(), entity_item.value(), tid, name,
                                offset, type_registry_, cid, &ptr);
}

gxf_result_t Runtime::GxfComponentFind(gxf_uid_t eid, void* item_ptr, gxf_tid_t tid,
                              const char* name, int32_t* offset, gxf_uid_t* cid, void** ptr) {
  EntityItem* entity_item = static_cast<EntityItem*>(item_ptr);
  return warden_->findComponent(context(), entity_item, tid, name, offset, type_registry_,
                                cid, ptr);
}

gxf_result_t Runtime::GxfComponentFindAll(gxf_uid_t eid, uint64_t* num_cids, gxf_uid_t* cids) {
  if (num_cids == nullptr) {
    GXF_LOG_ERROR("Buffer size was null when retrieving components for entity %05" PRId64 "", eid);
    return GXF_ARGUMENT_NULL;
  }
  if (cids == nullptr) {
    GXF_LOG_ERROR("Buffer was null when retrieving components for entity %05" PRId64 "", eid);
    return GXF_ARGUMENT_NULL;
  }

  const uint64_t max_cids = (*num_cids);

  const auto maybe_cids_vector = warden_->getEntityComponents(eid);
  if (!maybe_cids_vector) {
    GXF_LOG_ERROR("Failed to retrieve components for entity %05" PRId64 ": %s", eid,
                  GxfResultStr(maybe_cids_vector.error()));
    return ToResultCode(maybe_cids_vector);
  }

  const FixedVector<gxf_uid_t, kMaxComponents>& cids_vector = maybe_cids_vector.value();
  // Set the output variable to the number of components in the entity
  (*num_cids) = cids_vector.size();

  if (max_cids < cids_vector.size()) {
    GXF_LOG_ERROR(
        "Components buffer capacity %" PRIu64 ", but entity %05" PRId64 " contains %zu components",
        max_cids, eid, cids_vector.size());
    return GXF_QUERY_NOT_ENOUGH_CAPACITY;
  }

  std::copy(cids_vector.data(), cids_vector.data() + cids_vector.size(), cids);

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityGroupFindResources(gxf_uid_t eid, uint64_t* num_resource_cids,
                                                  gxf_uid_t* resource_cids) {
  if (num_resource_cids == nullptr) {
    GXF_LOG_ERROR("Buffer size was null when retrieving EntityGroup resource components "
                  "for entity %05" PRId64 "", eid);
    return GXF_ARGUMENT_NULL;
  }
  if (resource_cids == nullptr) {
    GXF_LOG_ERROR("Buffer was null when retrieving EntityGroup resource components "
                  "for entity %05" PRId64 "", eid);
    return GXF_ARGUMENT_NULL;
  }

  const uint64_t max_cids = (*num_resource_cids);

  const auto maybe_cids_vector = warden_->getEntityGroupResources(eid);
  if (!maybe_cids_vector) {
    GXF_LOG_ERROR(
        "Failed to retrieve EntityGroup resource components for entity %05" PRId64 ": %s",
        eid, GxfResultStr(maybe_cids_vector.error()));
    return ToResultCode(maybe_cids_vector);
  }

  const FixedVector<gxf_uid_t, kMaxComponents>& cids_vector = maybe_cids_vector.value();
  // Set the output variable to the number of components in the entity
  (*num_resource_cids) = cids_vector.size();

  if (max_cids < cids_vector.size()) {
    GXF_LOG_ERROR("Components buffer capacity %" PRIu64 ", but EntityGroup of entity %05" PRId64
                  " contains %zu resource components", max_cids, eid, cids_vector.size());
    return GXF_QUERY_NOT_ENOUGH_CAPACITY;
  }

  std::copy(cids_vector.data(), cids_vector.data() + cids_vector.size(), resource_cids);

  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfEntityGroupId(gxf_uid_t eid, gxf_uid_t* gid) {
  Expected<gxf_uid_t> result = warden_->entityFindEntityGroupId(eid);
  if (result) {
    *gid = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfEntityGroupName(gxf_uid_t eid, const char** name) {
  if (name == nullptr) {return GXF_NULL_POINTER; }
  const Expected<const char*> result = warden_->entityFindEntityGroupName(eid);
  if (result) {
    *name = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterSetFloat64(gxf_uid_t uid, const char* key, double value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %f", uid, key, value);
  return ToResultCode(parameters_->set<double>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetFloat32(gxf_uid_t uid, const char* key, float value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %f", uid, key, value);
  return ToResultCode(parameters_->set<float>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetInt8(gxf_uid_t uid, const char* key, int8_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRId8 "", uid, key, value);
  return ToResultCode(parameters_->set<int8_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetInt16(gxf_uid_t uid, const char* key, int16_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRId16 "", uid, key, value);
  return ToResultCode(parameters_->set<int16_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetInt32(gxf_uid_t uid, const char* key, int32_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRId32 "", uid, key, value);
  return ToResultCode(parameters_->set<int32_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetInt64(gxf_uid_t uid, const char* key, int64_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRId64 "", uid, key, value);
  return ToResultCode(parameters_->set<int64_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetUInt8(gxf_uid_t uid, const char* key, uint8_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRIu8 "", uid, key, value);
  return ToResultCode(parameters_->set<uint8_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetUInt16(gxf_uid_t uid, const char* key, uint16_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRIu16 "", uid, key, value);
  return ToResultCode(parameters_->set<uint16_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetUInt32(gxf_uid_t uid, const char* key, uint32_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRIu32 "", uid, key, value);
  return ToResultCode(parameters_->set<uint32_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetUInt64(gxf_uid_t uid, const char* key, uint64_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := %" PRIu64 "", uid, key, value);
  return ToResultCode(parameters_->set<uint64_t>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetStr(gxf_uid_t uid, const char* key, const char* value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := '%s'", uid, key, value);
  return ToResultCode(parameters_->setStr(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetPath(gxf_uid_t uid, const char* key, const char* value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := '%s'", uid, key, value);
  return ToResultCode(parameters_->setPath(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetHandle(gxf_uid_t uid, const char* key, gxf_uid_t value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s' := [C%05" PRId64 "]'", uid, key, value);
  return ToResultCode(parameters_->setHandle(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSetBool(gxf_uid_t uid, const char* key, bool value) {
  GXF_LOG_VERBOSE(
      "[C%05" PRId64 "] PROPERTY SET: '%s' := '%s'", uid, key, (value ? "true" : "false"));
  return ToResultCode(parameters_->set<bool>(uid, key, value));
}

gxf_result_t Runtime::GxfParameterSet1DVectorString(gxf_uid_t uid, const char* key,
                                                    const char* value[], uint64_t length) {
  if (!value) {
    GXF_LOG_ERROR("Value for the parameter, %s, is null", key);
    return GXF_ARGUMENT_NULL;
  }
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY SET: '%s'[0] := %s, ...", uid, key, value[0]);
  return ToResultCode(parameters_->setStrVector(uid, key, value, length));
}

gxf_result_t Runtime::GxfParameterSetFromYamlNode(gxf_uid_t uid, const char* key, void* yaml_node,
                                                  const char* prefix) {
  const YAML::Node& node = *static_cast<YAML::Node*>(yaml_node);
  const std::string prefix_str(prefix);
  return ToResultCode(parameters_->parse(uid, key, node, prefix_str));
}

gxf_result_t Runtime::GxfParameterGetAsYamlNode(gxf_uid_t uid, const char* key, void* yaml_node) {
  const Expected<YAML::Node> result = parameters_->wrap(uid, key);
  if (result) {
    YAML::Node* node = static_cast<YAML::Node*>(yaml_node);
    *node = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetFloat64(gxf_uid_t uid, const char* key, double* value) {
  const Expected<double> result = parameters_->get<double>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetFloat32(gxf_uid_t uid, const char* key, float* value) {
  const Expected<float> result = parameters_->get<float>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetInt64(gxf_uid_t uid, const char* key, int64_t* value) {
  const Expected<int64_t> result = parameters_->get<int64_t>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetUInt64(gxf_uid_t uid, const char* key, uint64_t* value) {
  const Expected<uint64_t> result = parameters_->get<uint64_t>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetUInt32(gxf_uid_t uid, const char* key, uint32_t* value) {
  const Expected<uint32_t> result = parameters_->get<uint32_t>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetUInt16(gxf_uid_t uid, const char* key, uint16_t* value) {
  const Expected<uint16_t> result = parameters_->get<uint16_t>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGet1DStrVector(gxf_uid_t uid, const char* key, char* value[],
                                                 uint64_t* count, uint64_t* min_length) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY GET: '%s'", uid, key);
  if (!value || !count || !min_length) { return GXF_ARGUMENT_NULL; }

  const Expected<std::vector<std::string>> result =
      parameters_->get<std::vector<std::string>>(uid, key);

  // calculate max size of stored strings
  uint64_t result_count = result.value().size();
  uint64_t result_max_length = 0;
  for (uint64_t i = 0; i < result_count; i++) {
    auto result_i_size = result.value()[i].size();
    result_max_length = (result_max_length < result_i_size) ? result_i_size : result_max_length;
  }

  // verify user has provided enough space
  if (result_count > *count || result_max_length > *min_length) {  // failure
    *count = result_count;
    *min_length = result_max_length;
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }

  for (uint64_t i = 0; i < result_count; i++) {
    auto result_str = result.value()[i];
    std::memcpy(value[i], result_str.data(), result_str.size());
  }
  *count = result_count;
  *min_length = result_max_length;
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfParameterInt64Add(gxf_uid_t uid, const char* key, int64_t delta,
                                           int64_t* value) {
  GXF_LOG_VERBOSE("[C%05" PRId64 "] PROPERTY ADD: '%s' + %" PRId64 "", uid, key, delta);

  const Expected<int64_t> result = parameters_->addGetInt64(uid, key, delta);
  if (result) {
    if (value != nullptr) {
      *value = result.value();
    }
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetStr(gxf_uid_t uid, const char* key, const char** value) {
  if (value == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  const Expected<const char*> result = parameters_->getStr(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    // Entity name is being searched using parameter storage, throw warning
    if (strcmp(kInternalNameParameterKey, key) == 0 && result.error() == GXF_PARAMETER_NOT_FOUND) {
      bool found = false;
      if (isSuccessful(GxfEntityIsValid(uid, &found)) && found == true) {
        GXF_LOG_WARNING("This API GxfParameterGetStr is deprecated for getting entity name."
        " Kindly use GxfEntityGetName api instead");
        return GxfEntityGetName(uid, value);
      }
    }
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetPath(gxf_uid_t uid, const char* key, const char** value) {
  if (value == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  const Expected<const char*> result = parameters_->getPath(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfParameterGetHandle(gxf_uid_t uid, const char* key, gxf_uid_t* value) {
  if (value == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  const auto maybe = parameters_->getHandle(uid, key);
  if (maybe) {
    *value = maybe.value();
    return GXF_SUCCESS;
  } else {
    return maybe.error();
  }
}

gxf_result_t Runtime::GxfEntityResourceGetHandle(gxf_uid_t eid, const char* type,
                        const char* resource_key, gxf_uid_t* resource_cid) {
  auto maybe_cid = ResourceManager::findEntityResourceByTypeName(context(),
                                    eid, type, resource_key);
  if (maybe_cid) {
    *resource_cid = maybe_cid.value();
    return GXF_SUCCESS;
  } else {
    return maybe_cid.error();
  }
}

gxf_result_t Runtime::GxfComponentType(gxf_uid_t cid, gxf_tid_t* tid) {
  const auto result = warden_->getComponentType(cid);
  if (!result) {
    return result.error();
  }
  *tid = result.value();
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfParameterGetBool(gxf_uid_t uid, const char* key, bool* value) {
  if (value == nullptr) {
    return GXF_ARGUMENT_NULL;
  }
  const auto maybe = parameters_->get<bool>(uid, key);
  if (maybe) {
    *value = maybe.value();
    return GXF_SUCCESS;
  } else {
    return maybe.error();
  }
}

gxf_result_t Runtime::GxfParameterGetInt32(gxf_uid_t uid, const char* key, int32_t* value) {
  const Expected<int32_t> result = parameters_->get<int32_t>(uid, key);
  if (result) {
    *value = result.value();
    return GXF_SUCCESS;
  } else {
    return result.error();
  }
}

gxf_result_t Runtime::GxfComponentPointer(gxf_uid_t uid, gxf_tid_t tid, void** pointer) {
  return shared_context_->findComponentPointer(context(), uid, pointer);
}

gxf_result_t Runtime::GxfGraphActivate() {
  const auto code = program_.activate();
  if (!code) {
    GXF_LOG_ERROR("Graph activation failed with error: %s", GxfResultStr(code.error()));
  }
  return ToResultCode(code);
}

gxf_result_t Runtime::GxfGraphRunAsync() {
  const auto code = program_.runAsync();
  if (!code) {
    GXF_LOG_ERROR("Graph run failed with error: %s", GxfResultStr(code.error()));
  }
  return ToResultCode(code);
}

gxf_result_t Runtime::GxfGraphInterrupt() {
  const auto code = program_.interrupt();
  if (!code) {
    GXF_LOG_ERROR("Graph interrupt failed with error: %s", GxfResultStr(code.error()));
  }
  return ToResultCode(code);
}

gxf_result_t Runtime::GxfGraphWait() {
  const auto code = program_.wait();
  if (!code) {
    GXF_LOG_ERROR("Graph wait failed with error: %s", GxfResultStr(code.error()));
  }
  return ToResultCode(code);
}

gxf_result_t Runtime::GxfGraphDeactivate() {
  const auto code = program_.deactivate();
  if (!code) {
    GXF_LOG_ERROR("Graph deactivation failed with error: %s", GxfResultStr(code.error()));
  }
  return ToResultCode(code);
}

gxf_result_t Runtime::GxfGraphRun() {
  gxf_result_t ret = GxfGraphRunAsync();
  if (ret != GXF_SUCCESS) {
    return ret;
  }

  return GxfGraphWait();
}

gxf_result_t Runtime::GxfLoadExtensionMetadataFiles(const char* const* filenames, uint32_t count) {
  if (filenames == nullptr) { return GXF_ARGUMENT_NULL; }
  for (uint32_t i = 0; i < count; ++i) {
    YAML::Node node;
    try {
      node = YAML::LoadFile(filenames[i]);
    } catch (YAML::Exception& e) {
      GXF_LOG_ERROR("Failed to load metadata yaml file '%s':\n%s", filenames[i], e.what());
      return GXF_FAILURE;
    }

    if (node[kAttributeComponents]) {
      const auto components = node[kAttributeComponents];
      if (!components.IsSequence() && !components.IsNull()) {
        GXF_LOG_ERROR("Components must be a sequence");
        return GXF_FAILURE;
      }

      for (size_t i = 0; i < components.size(); ++i) {
        YAML::Node comp = components[i];

        if (!components[i].IsMap()) {
          GXF_LOG_ERROR("Component metadata must be a map");
          return GXF_FAILURE;
        }
        if (!components[i][kAttributeTypeName]) {
          GXF_LOG_ERROR("Missing attribute \"typename\"");
          return GXF_FAILURE;
        }
        if (!components[i][kAttributeTypeId]) {
          GXF_LOG_ERROR("Missing attribute \"type_id\"");
          return GXF_FAILURE;
        }
        if (!components[i][kAttributeBaseTypeName]) {
          GXF_LOG_ERROR("Missing attribute \"base_typename\"");
          return GXF_FAILURE;
        }

        std::string type_name = components[i][kAttributeTypeName].as<std::string>();
        std::string type_id = components[i][kAttributeTypeId].as<std::string>();
        std::string base_typename = components[i][kAttributeBaseTypeName].as<std::string>();

        size_t pos = type_id.find("-");
        while (pos != std::string::npos) {
          type_id = type_id.replace(pos, 1, "");
          pos = type_id.find("-");
        }

        if (type_id.length() != 32) {
          GXF_LOG_ERROR("Invalid uuid : %s ", type_id.c_str());
          return GXF_FAILURE;
        }

        gxf_tid_t tid{std::stoul(type_id.substr(0, 16), nullptr, 16),
                      std::stoul(type_id.substr(16), nullptr, 16)};

        parameter_registrar_->addParameterlessType(tid, type_name);

        auto code = type_registry_->add(tid, type_name.c_str());
        if (!code) { return ToResultCode(code); }

        if (!base_typename.empty()) {
          code = type_registry_->add_base(type_name.c_str(), base_typename.c_str());
          if (!code) { return ToResultCode(code); }
        }
      }
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfSetSeverity(gxf_severity_t severity) {
  nvidia::Severity log_level;
  switch (severity) {
    case GXF_SEVERITY_NONE: {
      log_level = nvidia::Severity::NONE;
    } break;
    case GXF_SEVERITY_ERROR: {
      log_level = nvidia::Severity::ERROR;
    } break;
    case GXF_SEVERITY_WARNING: {
      log_level = nvidia::Severity::WARNING;
    } break;
    case GXF_SEVERITY_INFO: {
      log_level = nvidia::Severity::INFO;
    } break;
    case GXF_SEVERITY_DEBUG: {
      log_level = nvidia::Severity::DEBUG;
    } break;
    case GXF_SEVERITY_VERBOSE: {
      log_level = nvidia::Severity::VERBOSE;
    } break;
    default: {
      GXF_LOG_ERROR("Invalid severity level: %d", static_cast<int32_t>(severity));
      return GXF_FAILURE;
    } break;
  }

  nvidia::SetSeverity(log_level);
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfGetSeverity(gxf_severity_t* severity) {
  if (severity == nullptr) { return GXF_ARGUMENT_NULL; }
  const nvidia::Severity log_level = nvidia::GetSeverity();
  switch (log_level) {
    case nvidia::Severity::NONE: {
      *severity = GXF_SEVERITY_NONE;
    } break;
    case nvidia::Severity::ERROR: {
      *severity = GXF_SEVERITY_ERROR;
    } break;
    case nvidia::Severity::WARNING: {
      *severity = GXF_SEVERITY_WARNING;
    } break;
    case nvidia::Severity::INFO: {
      *severity = GXF_SEVERITY_INFO;
    } break;
    case nvidia::Severity::DEBUG: {
      *severity = GXF_SEVERITY_DEBUG;
    } break;
    case nvidia::Severity::VERBOSE: {
      *severity = GXF_SEVERITY_VERBOSE;
    } break;
    default: {
      GXF_LOG_ERROR("Invalid severity level: %d", static_cast<int32_t>(log_level));
      return GXF_FAILURE;
    } break;
  }
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfRedirectLog(FILE* fp) {
  nvidia::Redirect(fp, nvidia::Severity::ALL);
  return GXF_SUCCESS;
}

gxf_result_t Runtime::GxfSetExtensionLoader(ExtensionLoader* extension_loader) {
  if (extension_loader != nullptr) {
    extension_loader_ = extension_loader;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetEntityWarden(EntityWarden* warden) {
  if (warden != nullptr) {
    warden_ = warden;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetTypeRegistry(TypeRegistry* type_registry) {
  if (type_registry != nullptr) {
    type_registry_ = type_registry;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetParameterStorage(std::shared_ptr<ParameterStorage> parameters) {
  if (parameters != nullptr) {
    parameters_ = parameters;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetRegistrar(Registrar* registrar) {
  if (registrar != nullptr) {
    registrar_ = registrar;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetParameterRegistrar(
    ParameterRegistrar* parameter_registrar) {
  if (parameter_registrar != nullptr) {
    parameter_registrar_ = parameter_registrar;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetResourceRegistrar(
    std::shared_ptr<ResourceRegistrar> resource_registrar) {
  if (resource_registrar != nullptr) {
    resource_registrar_ = resource_registrar;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

gxf_result_t Runtime::GxfSetResourceManager(
    std::shared_ptr<ResourceManager> resource_manager) {
  if (resource_manager != nullptr) {
    resource_manager_ = resource_manager;
    return GXF_SUCCESS;
  }
  return GXF_NULL_POINTER;
}

}  // namespace gxf
}  // namespace nvidia
