# Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" Registry Core
"""

import errno
from glob import glob
from hashlib import new
import os, os.path as path
from os import listdir
from os.path import expanduser, abspath, isfile, join
from packaging import version
import platform
import shutil
import tempfile
import time
from typing import Dict
import yaml
import re

# fcntl is not available on windows
if platform.system() == "Linux":
    import fcntl

import registry.core.logger as log
from registry.bindings import pygxf
from registry.core.config import RegistryConfig
from registry.core.component import Component
from registry.core.database import DatabaseManager
from registry.core.dependency_governer import DependencyGovernor
from registry.core.extension import Extension
from registry.core.packager import Packager
from registry.core.repository import LocalRepository
from registry.core.repository_manager import RepositoryManager
from registry.core.utils import (PlatformConfig, ComputeConfig, TargetConfig, uuid_validator,
                                 ExtensionRecord, Variant, get_ext_subdir, target_to_str)
from registry.core.yaml_loader import YamlLoader
from registry.core.version import GXF_CORE_COMPATIBLE_VERSION, REGISTRY_CORE_VERSION
from registry.core.config import GRAPH_TARGETS_CONFIG

WINDOWS = platform.system() == "Windows"
logger = log.get_logger("Registry")
class RegistryCore:
    """ Registry Core class responsible for managing the deployment of
        extension across multiple users, targets and applications
    """

    def __init__(self):

        if not log.is_inited():
            log.init_logger()

        self._context = pygxf.gxf_context_create()
        if not self._context:
            self._gxf_core_version = GXF_CORE_COMPATIBLE_VERSION
        else:
            self._gxf_core_version = pygxf.get_runtime_version(self._context)
        self._yaml_loader = YamlLoader()
        self._exts: Dict[str: Extension] = {}   # for x86_64 variants
        self._cc_exts: Dict[str: Extension] = {} # for cross compiled variants
        self._global_components: Dict[str: Component] = {}
        self._registry_version = REGISTRY_CORE_VERSION

        # Setup registry config
        self._config = RegistryConfig()
        if not self._makedirs(self._cache_path()):
            raise OSError (errno.EIO,"Failed to create registry cache directory")

        # Setup registry lock
        self._lock = path.join(self._cache_path(), "gxf.registry.lock")
        self._lock_fd = None
        if platform.system() == "Linux" and not self._acquire_lock():
            raise OSError(errno.EIO, "Failed to acquire lock")

        # Setup utility helpers
        self._db = DatabaseManager(os.path.join(self._cache_path(), "gxf.db"))
        self._repo_mgr = RepositoryManager(self._config, self._db)
        self._dep_gov = DependencyGovernor(self._db)
        self._default_repo = LocalRepository("default", self._repo_path(), self._db)

        # Scan the default cache path for pre-existing cache info
        if not self._scan_cache():
            raise OSError(errno.EIO, "Cache is corrupted")

    def __del__(self):
        try:
            if not pygxf.gxf_context_destroy(self._context):
                logger.error("Failed to destroy Gxf context")
            self._release_lock()
        except AttributeError:
            # This can occur if constructor throws an error
            pass

    def registry_version(self):
        """ Fetch the current version of Registry

        Returns:
            str: current version of Registry
        """

        return self._registry_version

    def core_version(self):
        """ Fetch the version of Gxf Core currently supported by Registry

        Returns:
            str: version of Gxf Core currently supported by Registry
        """

        return self._gxf_core_version

    def add_extension(self, manifest_file: str, metadata_file: str):
        """ Adds an extension as specified using a manifest
            file. The manifest file lists extension metadata along
            with the path to the extension library.

        Args:
            manifest_file (str): Path to a manifest yaml file format
            metadata_file (str): Path to output metadata file
            reduced manifest yaml file format: -
            name : gst
            extension_library : bazel-out/k8-opt/bin/extensions/gst/libgxf_ext_gst.so
            uuid : 8ec2d5d6-b5df-48bf-8dee-0252606fdd7e
            version : 1.0.0
            license : MIT
            license_file : extensions/gst/LICENSE
            url : www.nvidia.com
            repository : https://github.com/NVIDIA
            .
            .
            .
        """
        if not metadata_file:
            metadata_file = "/tmp/gxf_registry_meta.yaml"

        manifest_file = self._cleanpath(manifest_file)
        manifest_node = self._yaml_loader.load_yaml(manifest_file)
        if not manifest_node:
            logger.error(f"Failed to load manifest {manifest_file}")
            return False

        # Load dependent extensions first and update the name string
        # since manifest uses extension bazel target name
        # Update the UUID's for extensions being loaded from NGC
        if "dependencies" in manifest_node:
            input_deps = manifest_node["dependencies"]
            output_deps = []
            for dep in input_deps:
                if "uuid" in dep:
                    dep["extension"] = self._db.get_extension_name(dep["uuid"])
                elif "extension" in dep:
                    dep["uuid"] = self._get_extension_uid(dep["extension"])
                else:
                    logger.error("Either UUID or extension must be specified for dependencies")
                    return False
                output_deps.append(dep)

            manifest_node["dependencies"] = output_deps

        if not Extension.validate_manifest(manifest_node):
            logger.error(f"Invalid extension manifest {manifest_file}")
            return False

        uuid = manifest_node["uuid"]
        platform = manifest_node["platform"]

        # Add / update local cache
        result = None
        if platform["arch"] == "x86_64":
            result = self._add_x86_extension(uuid, manifest_file, manifest_node)
        else:
            result = self._add_cc_extension(uuid, manifest_file, manifest_node)

        if not result:
            logger.error(f"Failed to add extension from manifest {manifest_file}")
            return False

        # Output the metadata to file
        return self._write_metadata(uuid, metadata_file)

    def _add_x86_extension(self, eid, manifest_file, manifest_node):
        lib_path = self._cleanpath(manifest_node["extension_library"])
        platform = manifest_node["platform"]
        compute = manifest_node["compute"]
        platform_cfg = PlatformConfig(platform["arch"], platform["os"], platform["distribution"])
        compute_cfg = ComputeConfig(compute["cuda"], compute["cudnn"], compute["tensorrt"],
                                    compute["deepstream"], compute["triton"], compute["vpi"])
        target_cfg = self._format_target_config(TargetConfig(platform_cfg, compute_cfg))

        # Load dependent extensions first
        if "dependencies" in manifest_node:
            input_deps = manifest_node["dependencies"]
            ext_records = []
            for dep in input_deps:
                record = ExtensionRecord(dep["extension"], dep["version"], dep["uuid"])
                ext_records.append(record)

            if not self._load_dependent_ext_metadata(eid, ext_records):
                logger.debug(f"Failed to load dependent extension metadata for {eid}")
                return False

        # Load new incoming extension
        if not pygxf.gxf_load_extensions(self._context, [lib_path]):
            logger.error(f"Failed to load extension {lib_path}")
            return False

        new_ext = Extension.from_gxf_core(self._context, eid, self._registry_version,
                                          lib_path, manifest_node)
        if not new_ext:
            logger.error(f"Failed to register extension {lib_path}")
            return False
        self._exts[eid] = new_ext

        # Update global component lookup
        for comp in self._exts[eid].components.values():
            comp_id = comp.cid
            self._global_components[comp_id] = comp

        # Update default repo
        if not self._add_to_default_repo(new_ext, lib_path, manifest_file, target_cfg):
            logger.error(f"Failed to add ext: {new_ext.name} to default repo")
            return False

        # Write metadata file
        e_node = new_ext.to_metadata()
        metadata_file = path.join(self._get_ext_dir_path(eid), "extension.yaml")
        with open(metadata_file, "w+") as f:
            yaml.dump(e_node, f, default_flow_style=False, sort_keys=False)

        # Remove any older versions of the extension in db corresponding to default repo
        versions = self._db.get_ext_versions_from_repo("default", new_ext.name, new_ext.uuid)
        if versions:
            for ver in versions:
                if new_ext.version != ver:
                    e = ExtensionRecord(new_ext.name, ver, new_ext.uuid)
                    if not self._db.remove_extension_from_database(e):
                        return False

        if versions and new_ext.version in versions:
            e = ExtensionRecord(new_ext.name, new_ext.version, new_ext.uuid)
            cur_metadata = self._db.get_interface_metadata(e)
            # Check if interface is changed and force update by removing
            # existing entry and re-adding from metadata
            if cur_metadata != e_node:
                logger.info(f"Interface for ext: {new_ext.name} updated. Removing existing interface and variants from default repo")
                if not self._db.remove_extension_from_database(e):
                    return False
                versions = []

        if not versions or new_ext.version not in versions:
            # Add interface to db
            if not self._db.add_interface_to_repo("default", metadata_file):
                return False

        # Add variant to db
        variant = Variant(gxf_core_version=self._gxf_core_version,
                          registry_version=self._registry_version,
                          target = target_cfg,
                          hash=None)
        if not self._db.add_variant(eid, new_ext.version, "default", variant):
            return False

        return True

    def _add_cc_extension(self, eid, manifest_file, manifest_node):
        # Check if x86 variant is already registered
        if not eid in self._exts.keys():
            logger.error("x86_64 variant of the extension should be registered first")
            return False
        new_ext = Extension.from_manifest(manifest_node, self._registry_version)
        if not new_ext:
            logger.error("Invalid manifest for non x86_64 variant extension")
            return False
        self._cc_exts[eid] = new_ext

        lib_path = self._cleanpath(new_ext.extension_library)
        platform_cfg = PlatformConfig(new_ext.arch, new_ext.os, new_ext.distribution)
        compute_cfg = ComputeConfig(new_ext.cuda, new_ext.cudnn,new_ext.tensorrt,
                                    new_ext.deepstream, new_ext.triton, new_ext.vpi)
        target_cfg = self._format_target_config(TargetConfig(platform_cfg, compute_cfg))

        if not self._add_to_default_repo(new_ext, lib_path, manifest_file, target_cfg):
            logger.error(f"Failed to add ext: {new_ext.name} to default repo")
            return False

        variant = Variant(gxf_core_version=self._gxf_core_version,
                          registry_version=self._registry_version,
                          target=target_cfg,
                          hash=None)

        return self._db.add_variant(new_ext.uuid, new_ext.version, "default", variant)

    def import_extension_interface(self, ext_name, version: str, import_dp, uuid = None):
        """Imports an extension interface from an external repo(ngc) to a local directory
           using extension uuid

        Args:
            ext_name (str): extension name
            version (str): required version of the extension
            import_dp (str): directory to import extension file contents
            uuid (str): uuid of the extension
        """
        logger.info("Importing extension interface ...")
        # Use uuid whenever available
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(ext_name)
            if not uuid:
                return False

        ext = self._ext_lookup(uuid)
        if not ext:
            logger.error(f"Extension not found. Did you forget to add or sync external"
                                " repository ?")
            return False
        ext_rec = ExtensionRecord(ext_name, version, uuid)
        ext_repo = self._db.get_extension_source_repo(ext_rec)
        if not ext_repo:
            logger.error(f"Could not find extension {ext_name} version {version} in cache")
            return False
        if ext_repo == "default":
            logger.error(f"Extension found in default repository.")
            return False

        return self._repo_mgr.import_extension_interface(ext_repo, import_dp, uuid, version)

    def import_extension_variant(self, ext_name: str, version: str, target_cfg: TargetConfig,
                                 import_dp: str, uuid = None):
        """Imports an extension from an external repo(ngc) to a local directory
           using extension uuid

        Args:
            ext_name (str): extension name
            version (str): required version of the extension
            target_cfg (TargetConfig): target platform spec required for import
            import_dp (str): directory to import extension file contents
            uuid (str): uuid of the extension
        """
        logger.info("Importing extension variant ... ")
        # Use uuid whenever available
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(ext_name)
            if not uuid:
                return False

        ext = self._ext_lookup(uuid)
        if not ext:
            logger.error(f"Extension not found. Did you forget to add or sync external"
                                " repository ?")
            return False

        target_cfg = self._format_target_config(target_cfg)
        if not target_cfg:
            logger.error("Invalid target configuration")
            return False

        ext_rec = ExtensionRecord(ext_name, version, uuid)
        ext_repo = self._db.get_extension_source_repo(ext_rec)
        if not ext_repo:
            logger.error(f"Could not find extension {ext_name} version {version} in cache")
            return False
        if ext_repo == "default":
            logger.error(f"Extension found in default repository.")
            return False

        return self._repo_mgr.import_extension_variant(ext_repo, import_dp, uuid, version, target_cfg)

    def remove_extension_interface(self, ext_name: str, version: str, repo: str, uuid = None):
        """Remove an extension interface from a repository

        Args:
            ext_name (str): extension name
            uuid (str): uuid of the extension
            version (str): required version of the extension to be removed
            repo (str): name of the repository from which the extension has to be removed
        """
        logger.info("Removing extension interface ...")
        ext = None
        # Use uuid whenever available
        if uuid:
            if not uuid_validator(uuid):
                return False

            ext = self._ext_lookup(uuid)
            if not ext:
                logger.error(f"Extension not found. Did you forget to add or sync external"
                                    " repository ?")
                return False
            ext_name = ext.name
        else:
            uuid = self._get_extension_uid(ext_name)
            if not uuid:
                return False
            ext = self._ext_lookup(uuid)

        return self._repo_mgr.remove_extension_interface(repo, ext_name, uuid, version)

    def remove_extension_variant(self, ext_name: str, version: str,
                                repo_name: str, target_cfg: TargetConfig,
                                uuid: str):
        """Remove an extension variant from a repository

        Args:
            ext_name (str): extension name
            version (str): required version of the extension
            repo_name (str): name of the repository from which the extension has to be removed
            target_cfg (TargetConfig): target platform spec required for import
            uuid (str): uuid of the extension
        """
        logger.info("Removing extension variant ...")
        ext = None
        # Use uuid whenever available
        if uuid:
            if not uuid_validator(uuid):
                return False

            ext = self._ext_lookup(uuid)
            if not ext:
                logger.error(f"Extension not found. Did you forget to add or sync external"
                                    " repository ?")
                return False
            ext_name = ext.name
        else:
            uuid = self._get_extension_uid(ext_name)
            if not uuid:
                return False

        target_cfg = self._format_target_config(target_cfg)
        return self._repo_mgr.remove_extension_variant(repo_name, ext_name, uuid, version, target_cfg)

    def remove_extension(self, ext_name: str, version: str, repo_name = None):
        """Remove an extension from a repository

        Args:
            ext_name (str): extension name
            version (str): required version of the extension
            repo_name (str): name of the repository from which the extension has to be removed
        """
        uuid = self._get_extension_uid(ext_name)
        ext = self._ext_lookup(uuid)
        if not ext:
            logger.error(f"Extension not found. Did you forget to add or sync external"
                                " repository ?")
            return False

        if not repo_name:
            repo_name = self._db.get_extension_source_repo(ExtensionRecord(ext_name, version, uuid))

        variants = self.get_extension_variants(ext.name, ext.version, repo_name, ext.uuid)
        if variants is None:
            logger.warning(f"Failed to fetch variants for extension: {ext.name}")
            return False

        for var in variants:
            if not self.remove_extension_variant(ext.name, ext.version, repo_name, var.target, ext.uuid):
                logger.warning(f"Failed to remove variant for extension: {ext.name} "
                                     f"variant: {target_to_str(var.target)}")

        if not self.remove_extension_interface(ext.name, ext.version, repo_name, ext.uuid):
            logger.warning(f"Failed to remove interface for extension:{ext.name}")

        return True

    def remove_all_extensions(self, repo_name: str):
        logger.info("Removing all extensions ...")

        if not self._repo_mgr.does_repo_exist(repo_name):
            return False

        exts = self._db.get_extensions_from_repo(repo_name)
        if exts is None:
            return False

        if len(exts) == 0:
            logger.error(f"No extensions found in {repo_name} repository")
            return False

        for ext in exts:
            self.remove_extension(ext.name, ext.version, repo_name)

        return True

    def _cleanup(self):
        """ Delete all the current cache specific data stored in the registry
        """

        self._exts = {}
        self._cc_exts = {}
        self._global_components = {}

    def _scan_cache(self):
        """ Scan the Registry Cache for Extensions. This action can be
            reset using `~core.core._cleanup`
        """

        self._cleanup()

        interfaces = self._db.get_frontend_interfaces()
        if interfaces is None:
            return False

        for i in interfaces:
            eid = i["uuid"]
            cur_ext = Extension.from_metadata(i)
            if cur_ext is None:
                logger.error("Cache is corrupted")
                return False

            self._exts[eid] = cur_ext

            # Maintain global lookup for all components
            for comp in self._exts[eid].components.values():
                comp_id = comp.cid
                self._global_components[comp_id] = comp

        return True

    def _add_to_default_repo(self, ext: Extension, ext_lib_path: str, manifest_path: str,
                             target_cfg: TargetConfig):
        """ Copy extension files like extension library, headers and data
            files to default repo
        """
        eid = ext.uuid
        ext_subdir = get_ext_subdir(eid, ext.version, target_cfg)
        ext_subdir_path = self._get_ext_subdir_path(eid, ext.version, target_cfg)

        # Remove variant from default repo if already present
        if path.exists(ext_subdir_path):
            self._default_repo.remove_from_repo(ext_subdir)

        # Copy ext lib
        res = self._default_repo.add_to_repo(ext_lib_path, ext_subdir)
        if not res:
            logger.error(f"Failed to add \"{ext_lib_path}\" to default repo")
            return False

        # Copy ext manifest
        res = self._default_repo.add_to_repo(manifest_path, ext_subdir, "manifest.yaml")
        if not res:
            logger.error(f"Failed to add \"{manifest_path}\" to default repo")
            return False

        # Create a target.yaml corresponding to the target configuration
        target_node = ext.to_target()
        target_file = path.join(ext_subdir_path, "target.yaml")
        with open(target_file, "w+") as fd:
            yaml.dump(target_node, fd, default_flow_style=False, sort_keys=False)

        def add_file_list(files, path):
            for f in files:
                res = self._default_repo.add_to_repo(f, path)
                if not res:
                    logger.error(f"Failed to add {f} to default repo")
                    return False
            return True

        # Add headers only for x86 variant since it's an interface to the extension
        if target_cfg.platform.arch == "x86_64":
            if not add_file_list(ext.headers, path.join(eid,"headers")):
                return False

            if not add_file_list(ext.python_sources, path.join(eid, "py_srcs")):
                return False

            # LICENSE file is currently optional, copy if present
            if ext.license_file and path.exists(ext.license_file):
                if not self._default_repo.add_to_repo(ext.license_file, eid):
                    logger.error(f"Failed to add \"{ext.license_file}\" to default repo")
                    return False

        if not add_file_list(ext.binaries, ext_subdir):
            return False
        if not add_file_list(ext.python_bindings, ext_subdir):
            return False
        if not add_file_list(ext.data, ext_subdir):
            return False

        return True

    def _write_metadata(self, eid: str, meta_file: str):
        """ Dumps the extension metadata into path specified using meta_file
            To be used only with add_extension(...) with one extension
        """

        e_node = self._exts[eid].to_metadata()
        if not e_node:
            name = self._get_extension_name(eid)
            name = name if name else ""
            logger.error(f"Missing extension {name}")
            logger.debug(f"Missing extension {eid}")
            return False

        with open(meta_file, "w+") as meta:
            yaml.dump(e_node, meta, default_flow_style=False, sort_keys=False)

        return True

    def _load_dependent_ext_metadata(self, ext_id, deps):
        """ Load all required dependent extensions before loading a new extension

        Args:
            ext_list: list of eid of the extensions that need to be loaded
            deps_graph Dict{str:[str]}: list of preidentified dependencies which
                                        are not in cache
        """

        logger.debug(f"Loading dependent extensions for {ext_id}")
        deps = self._dep_gov.find_all_dependencies(deps)
        if deps is None:
            return False
        if len(deps) == 0:
            return True

        dep_exts = []
        for dep in deps:
            meta_node = self._db.get_interface_metadata(dep)
            cache_path = self._makedirs(self._get_ext_cache_path(dep.uuid))
            ext_meta_path = path.join(cache_path, "metadata.yaml")

            with open(ext_meta_path, "w") as f:
                yaml.dump(meta_node, f, default_flow_style=False, sort_keys=False)

            logger.debug(f"Loading ext metadata: {ext_meta_path}")
            dep_exts.append(ext_meta_path)

        # Load all extensions to gxf core context
        return pygxf.gxf_load_extension_metadata(self._context, dep_exts)

    def set_cache_path(self, cache_path: str):
        """ Update the path for the cache used by Registry

        Args:
            cache_path (str): New path for Registry Cache

        Returns:
            bool : Returns true is the cache path was successfully updated
        """
        if not cache_path:
            logger.error("cache path cannot be empty")
            return False

        new_cache_path = self._cleanpath(cache_path + "/gxf_registry/")

        if os.path.exists(new_cache_path) and not os.path.isdir(new_cache_path):
            logger.error(f"Path {new_cache_path} exists and is not a directory, please remove it")
            return False
        res = self._makedirs(new_cache_path)
        if not res:
            logger.error(f"Failed to set new cache path {new_cache_path}")
            return False

        self._config.set_cache_path(new_cache_path)
        return True

    def get_cache_path(self):
        """ Get the path for the cache used by Registry

        Returns:
            str: Path for Registry Cache
        """

        return self._config.get_cache_path()

    def clear_cache(self):
        """ Delete all extension libs and metadata stored in cache path
        """
        cache_path = self._config.get_cache_path()
        self._db.drop_all_tables()
        self._db.close_connection()
        try:
            shutil.rmtree(cache_path)
            return True
        except IOError:
            logger.error(f"Failed to remove cache path {cache_path}")
            return False

    def refresh_cache(self):
        """ Refresh cache content. Fetch the latest version of all the extensions
            in cache from their corresponding source repositories
        """
        logger.info("Refreshing cache ...")

        for eid in self._exts.keys():
            repo = self._ext_repo_lookup(eid)
            res = self._repo_mgr.refresh_extension(repo, eid)
            if not res:
                name = self._get_extension_name(eid)
                logger.error(f"Failed to sync {name} from {repo}")
                return False

        return True

    def clean_default_repo(self):
        """ Cleans the default repo content
        """

        return self._db.remove_all_extensions("default") and self._default_repo.clean()

    def fetch_extension_list(self, repo_name=None):
        """ Fetch list of all registered extensions

        Returns:
            List[Extension_Obj] : Returns a list of Extension info
                                     objects
        """

        result = [val.to_tuple() for val in self._exts.values()]
        if repo_name:
            result = [elm for elm in result
                      if self._ext_repo_lookup(elm.id) == repo_name]
        return result

    def fetch_extension(self, eid: str):
        """ Fetch an extension using eid

        Args:
            eid (str): uuid of extension

        Returns:
            Extension_Obj: Tuple containing extension info
        """

        if eid not in self._exts.keys():
            return None

        else:
            result = self._exts[eid].to_tuple()
            return result

    def sync_extension(self, repo_name, ext_name, version, uuid):
        # Use uuid whenever available
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(ext_name)
            if not uuid:
                return False

        return self._repo_mgr.sync_extension(repo_name, uuid, version) and self._scan_cache()

    def set_extension_frontend_version(self, ext_name, version, uuid=None):
        # Use uuid whenever available
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(ext_name)
            if not uuid:
                return False

        versions_map = self.get_extension_versions(ext_name, uuid)
        for vers in versions_map.values():
            for v in vers:
                if v == version:
                    return self._db.update_frontend_flag(uuid, version)

        logger.error(f"Failed to find version {version} for extension {ext_name}")
        return False

    def fetch_component_list(self, eid: str):
        """ Fetch a list component using extension uuid

        Args:
            eid (str): uuid of extension

        Returns:
            List[Component_Obj]: List of tuple containing component info
        """

        if eid not in self._exts.keys():
            return None

        else:
            comps = self._exts[eid].fetch_component_list()
            result = [comp.to_tuple() for comp in comps]
            return result

    def fetch_component(self, cid: str):
        """ Fetch a component using component uuid

        Args:
            cid (str): uuid of component

        Returns:
            Component_Obj: Tuple containing component info
        """

        if cid not in self._global_components.keys():
            return None

        else:
            result = self._global_components[cid].to_tuple()
            return result

    def fetch_parameter_list(self, cid: str):
        """ Fetch a list of parameters using component uuid

        Args:
            cid (str): uuid of component

        Returns:
            List[Parameter_Obj]: List of tuple containing parameter info
        """

        if cid not in self._global_components.keys():
            return None

        else:
            comp = self._global_components[cid]
            params = comp.fetch_parameter_list()
            result = [p.to_tuple() for p in params]
            return result

    def get_extension_source_repo(self, extn_name, version, uuid = None):
        if uuid and not uuid_validator(uuid):
              return None
        else:
            uuid = self._get_extension_uid(extn_name)
            if not uuid:
                return None

        ext_rec = ExtensionRecord(extn_name, version, uuid)
        return self._db.get_extension_source_repo(ext_rec)

    def get_extension_uuid(self, extn_name):
        res = self._get_extension_uid(extn_name)
        if not res:
            logger.error(f"Coudldn't find the uuid for extension {extn_name}")
        return res

    def get_extension_versions(self, extn_name, uuid=None):
        """Fetch all the versions of an extension found in it's source repository

        Args:
            extn_name (str): extension name
            uuid (str): uuid of the extension
            repo_name (str): name of the repository from which the extension has to be removed
        """
        logger.debug("Getting extension versions ...")
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(extn_name)
            if not uuid:
                return False

        return self._db.get_all_ext_versions(extn_name, uuid)

    def get_extension_variants(self, extn_name, version, repo_name=None, uuid=None):
        """Fetch all the variants of an extension found in it's source repository

        Args:
            extn_name (str): extension name
            version(std): extension version
            uuid (str): uuid of the extension
            repo_name (str): name of the repository from which the extension has to be removed
        """
        logger.debug("Getting extension variants ...")
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(extn_name)
            if not uuid:
                return False
        if not repo_name:
            repo_name = self._db.get_extension_source_repo(ExtensionRecord(extn_name, version, uuid))

        return self._db.get_variants(uuid, version, repo_name)

    def get_extension_dependencies(self, extn_name, version, uuid=None):
        """Fetch all the dependencies of an extension

        Args:
            extn_name (str): extension name
            version(std): extension version
            uuid (str): uuid of the extension
            repo_name (str): name of the repository from which the extension has to be removed
        """
        logger.debug("Getting extension dependencies ...")
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(extn_name)
            if not uuid:
                return False

        ext_rec = ExtensionRecord(extn_name, version, uuid)
        deps = self._dep_gov.find_all_dependencies([ext_rec])
        if not deps:
            return False

        # Remove src extension which is also added as a dependency
        result  = [dep for dep in deps if dep.uuid != uuid]
        return result

    def update_graph_dependencies(self, graphs: list):
        """ Update graph to the latest version of its dependencies.
        """
        logger.info("Updating graph dependencies...")
        latest_ext_versions = {}
        for ext in self._exts.values():
            latest_ext_versions[ext.uuid] = self._db.get_latest_ext_version(ext.name, ext.uuid)

        result = True
        for graph in graphs:
            graph = self._cleanpath(graph)
            docs = self._yaml_loader.load_all(graph)
            if not docs:
                logger.warning(f"Failed to load graph {graph}")
                result = False
                continue

            graph_deps = None
            for i in range(len(docs)):
                if docs[i].get("dependencies"):
                    graph_deps = docs[i].get("dependencies")
                    updated_deps = []

                    for dep in graph_deps:
                        uuid = dep["uuid"]
                        ver = dep["version"]
                        name = dep["extension"]
                        if uuid not in latest_ext_versions:
                            logger.warning(f"Extension: {name} not found in cache. Version not updated")
                            updated_deps.append(dep)
                            continue

                        if version.parse(ver) < version.parse(latest_ext_versions[uuid]):
                            dep["version"] = latest_ext_versions[uuid]

                        updated_deps.append(dep)

                    # Replace the updated list of dependencies in the original yaml content
                    docs[i]["dependencies"] = updated_deps

            # Save graph file
            with open(graph, "w+") as f:
                yaml.dump_all(docs, f, default_flow_style=False, sort_keys=False)

        return result

    def install_extension(self, ext_name, target_fp, ext_ver):
        uuid = self._get_extension_uid(ext_name)
        if not uuid:
            return False

        # Check if version matches frontend, query from db if its not a match
        ext = self._exts[uuid]
        if not ext_ver:
            ext_ver = ext.version
        elif ext_ver != ext.version:
            ext_meta = self._db.get_interface_metadata(ExtensionRecord(ext_name, ext_ver, uuid))
            ext = Extension.from_metadata(ext_meta)

        exts = self._dep_gov.find_all_dependencies([ExtensionRecord(ext_name, ext_ver, uuid)])
        if not exts:
            return None

        # use current platform target config if not specified
        if not target_fp:
            target_fp = self._config.get_platform_config()

        target_cfg = self._read_graph_target_file(target_fp)
        if not target_cfg:
            return None

        variants = self._dep_gov.get_best_variants(exts, target_cfg)
        workspace_root = self._config.get_workspace_root()

        for e, target_cfg in variants.items():
            ext = self._ext_lookup(e.uuid)
            ext_namespace = "gxf/"
            if ext.namespace:
                ext_namespace = ext.namespace
            py_alias = ext.python_alias if ext.python_alias else ext.name
            dst_path = path.normpath(path.join(workspace_root, ext_namespace, py_alias))
            if path.exists(dst_path):
                shutil.rmtree(dst_path)
            self._makedirs(dst_path)

            ext_repo = self._db.get_extension_source_repo(ext)
            if ext_repo != "default":
                if ext.python_sources:
                    logger.debug(f"Importing py srcs from ngc repo for {ext.name}")
                    result = self._repo_mgr.import_extension_py_srcs(ext_repo, dst_path, ext.uuid,
                                                         ext.version, ext.name)
                    if not result:
                        return None
                logger.debug(f"Importing ext libs from ngc repo for {ext.name}")
                result = self._repo_mgr.import_extension_variant(ext_repo, dst_path, ext.uuid,
                                                     ext.version, target_cfg)
                if not result:
                    return None

            else:
                if ext.python_sources:
                    py_srcs = path.join(self._get_ext_dir_path(ext.uuid), "py_srcs")
                    shutil.copytree(py_srcs, dst_path, dirs_exist_ok=True)
                else:
                    logger.debug(f"Extension {e.name}:{e.version} doesn't have any py srcs")

                ext_subdir_path = self._get_ext_subdir_path(ext.uuid, ext.version, target_cfg)
                if not path.isdir(ext_subdir_path):
                    logger.error(f"Missing extension variant in default repo {ext_subdir_path}")
                    return None
                shutil.copytree(ext_subdir_path, dst_path, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("manifest.yaml","target.yaml"))

        return True

    def install_graph(self, graphs: list, target_cfg: TargetConfig):
        """ Install extensions required for graph and return manifest

        Args:
            graph (str): Graph file path
            target_cfg(TargetConfig): Platform config

        Returns:
            list(str): List of dependent extensions required to
                       execute the graph
        """
        logger.info("Installing graph ...")

        deps_list = []
        # Parse all the graphs and collect the dependencies
        for graph in graphs:
            graph = self._cleanpath(graph)
            docs = self._yaml_loader.load_all(graph)
            if not docs:
                logger.error(f"Failed to load graph {graph}")
                return None

            docs = list(filter(lambda x: x is not None, docs))
            graph_deps = [doc.get("dependencies") for doc in docs]
            if not graph_deps:
                logger.error("No extension dependencies found")
                return None

            graph_deps = list(filter(lambda x: x is not None, graph_deps))
            graph_deps_flat = [entry for dep in graph_deps for entry in dep]
            deps_list.extend(graph_deps_flat)

        if not Extension.validate_dependencies(deps_list):
            return None

        graph_uuids = set()
        graph_exts = []
        for dep in deps_list:
            if dep["uuid"] is not graph_uuids:
                graph_uuids.add(dep["uuid"])
                graph_exts.append(ExtensionRecord(dep["extension"], dep["version"], dep["uuid"]))
            else:
                logger.warning(f"Duplicate extension found in dependencies {dep['extension']}:{dep['version']}")

        if not graph_exts:
            logger.error(f"Failed to find any extensions used in graph {graph}")
            return None

        logger.debug(f"List of all extensions dependencies needed: \n {graph_exts}")
        exts = self._dep_gov.find_all_dependencies(graph_exts)
        if not exts:
            return None

        return self._dep_gov.get_best_variants(exts, target_cfg)

    def install_graph_with_archive(self, graph_paths: list,
                                         manifest_path: str,
                                         archive_dirpath: str,
                                         in_export_dp: str,
                                         target_filepath: str):
        """ Install extensions required for graph_path and create an equivalent
            manifest with the graph package contents exported an archive

        Args:
            graph_path (str): Graph file path
            manifest_path (str): Manifest file path
            archive_dirpath (str): Tarball path
            in_export_dp (str): directory structure to be used inside the archive
            target_filepath(str): Graph target file path

        Returns:
            bool: True if graph installation is successful
        """

        # Default path for extension contents inside an archive
        in_export_dp = in_export_dp if in_export_dp is not None else "/opt/nvidia/gxf/"

        # Temporary staging area to build an archive
        staging_path =  tempfile.mkdtemp(prefix="/tmp/gxf.")

        # Install the graph to a staging area
        if not self.install_graph_with_dir(graph_paths, manifest_path, staging_path, in_export_dp, target_filepath, True):
            return None

        # Create archive
        graph_name = graph_paths[0].split("/")[-1].split(".yaml")[0]
        pkg = Packager(archive_dirpath, graph_name)
        pkg.addDirectory(staging_path)
        pkg.zip()

        # Cleanup staging area
        try:
            shutil.rmtree(staging_path)
        except IOError:
            logger.error(f"Failed to remove {staging_path}")
            return False


        return pkg.package_path

    def install_graph_with_dir(self, graph_paths: str,
                                     manifest_path: str,
                                     output_path: str,
                                     in_export_dp: str,
                                     target_filepath: str,
                                     use_archive_path: bool = False):
        """ Install extensions required for graph_path and create an equivalent
            manifest with the graph package contents exported an output directory

        Args:
            graph_path (str): Graph file path
            manifest_path (str): Manifest file path
            output_path: Output directory path to copy the contents
            target_filepath(str): Graph target file path
                                - might be file path or target key ['x86', 'aarch']
                                  or target config (yaml) itself

        Returns:
            bool: True if graph installation is successful
        """
        # Use a tmp staging area to download all ngc extensions
        staging_path = tempfile.mkdtemp(prefix= tempfile.gettempdir() + "//.gxf.")
        logger.debug(f"Installing graph in dir {staging_path}")

        target_cfg = None
        # if target path is specified
        if os.path.exists(os.path.abspath(target_filepath)):
            target_cfg = self._read_graph_target_file(target_filepath)
        # if target cfg is directly passed
        elif isinstance(target_filepath, TargetConfig):
            target_cfg = target_filepath
        # if target key ('x86','aarch') is passed
        else:
            target_cfg = self._get_target_config(target_filepath)

        if not target_cfg:
            logger.error(f"Failed to get any target configs from {target_filepath}."\
                         " Please pass the target keyword - x86 / aarch64")
            return None

        manifest_exts = self.install_graph(graph_paths, target_cfg)
        if not manifest_exts:
            logger.error(f"Failed to find extensions for manifest from graphs {graph_paths}")
            return None

        manifest_libs = []

        for e, target_cfg in manifest_exts.items():
            ext_subdir_path = None
            ext_repo = self._db.get_extension_source_repo(e)
            ext_meta = self._db.get_interface_metadata(e)
            ext = Extension.from_metadata(ext_meta)

            if ext_repo == "default":
                ext_subdir_path = self._get_ext_subdir_path(ext.uuid, ext.version, target_cfg)
                if not path.isdir(ext_subdir_path):
                    logger.error(f"Missing extension variant in default repo {ext_subdir_path}")
                    return None

            # Import extension if its not present in default repo
            else:
                logger.debug(f"Importing ext libs from ngc repo for {ext.name}")
                ext_subdir_path = path.join(staging_path, ext.uuid)
                result = self._repo_mgr.import_extension_variant(ext_repo, ext_subdir_path, ext.uuid,
                                                         ext.version, target_cfg)
                if not result:
                    return None

            # adding a directory prefix to have consistent python bindings
            # in the src and installed dir

            if "namespace" in ext_meta.keys() and ext_meta['namespace']:
                directory_prefix = ext_meta['namespace']+'/'
            else:
                directory_prefix = 'gxf/'
            new_extension_name = ""
            if "python_alias" in ext_meta.keys() and ext_meta['python_alias']:
                new_extension_name = ext_meta['python_alias']
            else:
                new_extension_name = ext.name
            in_arc_path = path.normpath(path.join(output_path + "/" + in_export_dp, directory_prefix, new_extension_name))
            try:
                if path.exists(in_arc_path):
                    shutil.rmtree(in_arc_path)

                # Populate ext variant libs in output path
                shutil.copytree(ext_subdir_path, in_arc_path, ignore=shutil.ignore_patterns("manifest.yaml","target.yaml"))

            except shutil.Error as e:
                logger.error(e)
                logger.error(f"Failed to add extension variant {ext_subdir_path}")
                return None

            try:
                init_file_path = output_path + "/" + in_export_dp + directory_prefix + "__init__.py"
                with open(init_file_path, "w") as f:
                    f.write("\n"
                    "# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.\n"
                    "#\n"
                    "# NVIDIA CORPORATION and its licensors retain all intellectual property\n"
                    "# and proprietary rights in and to this software, related documentation\n"
                    "# and any modifications thereto.  Any use, reproduction, disclosure or\n"
                    "# distribution of this software and related documentation without an express\n"
                    "# license agreement from NVIDIA CORPORATION is strictly prohibited.\n")
            except Exception as e:
                logger.error(e)
                return None

            # Add ext lib to manifest
            iface = self._db.get_interface_metadata(ext)
            e = Extension.from_metadata(iface)
            if use_archive_path:
                manifest_libs.append(path.join(in_export_dp, directory_prefix, new_extension_name, e.extension_library))
            else:
                manifest_libs.append(path.join(output_path, in_export_dp, directory_prefix, new_extension_name, e.extension_library))
        # Write the manifest file
        try:
            f = open(manifest_path, "w+")
        except IOError:
            logger.error(f"Cannot open {manifest_path} for writing")
            return False
        finally:
            if WINDOWS:
                manifest_libs = [path.normpath(os.path.relpath(lib, output_path)).replace("\\", "/") for lib in manifest_libs]
            else:
                manifest_libs = [path.normpath(lib) for lib in manifest_libs]
            manifest_yaml = {"extensions" : manifest_libs, "variants": self._exts_to_map(manifest_exts)}
            yaml.dump(manifest_yaml, f, default_flow_style=False, sort_keys=False)
            f.close()

        # Cleanup staging area
        try:
            shutil.rmtree(staging_path)
        except OSError:
            logger.warning(f"Failed to remove {staging_path}")
            return False

        logger.debug(f"Graph successfully installed to {output_path}")
        return True

    def _ext_lookup(self, eid : str):
        if eid not in self._exts.keys():
            return None

        return self._exts[eid]

    def _ext_repo_lookup(self, eid: str):
        if eid not in self._exts.keys():
            logger.error(f"Extension not found in cache.")
            logger.debug(f"Extension {eid} not found in cache.")
            return None

        ext = self._exts[eid]
        query = ExtensionRecord(ext.name, ext.version, ext.uuid)
        return self._db.get_extension_source_repo(query)

    def _get_extension_uid(self, name: str):
        result = []
        for ext in self._exts.values():
            if ext.name == name:
                result.append(ext.uuid)
        if len(result) == 0:
            logger.error(f"Failed to find extension: {name}")
            return None
        if len(result) > 1:
            logger.error(f"Multiple extensions found with name {name}")
            return None
        return result[0]

    def _get_extension_name(self, uid: str):
        for ext in self._exts.keys():
            if uid == ext:
                return self._exts[uid].name
        return None

    def publish_all_extensions(self, repo_name: str, force=False):
        logger.info("Publishing all extensions from default repository")

        if not self._repo_mgr.does_repo_exist(repo_name):
            return False

        exts = self._db.get_extensions_from_repo("default")
        if exts is None:
            return False

        if len(exts) == 0:
            logger.error("No extensions found in default repository")
            return False

        for ext in exts:
            self.publish_extension(ext.name, repo_name, force)

        return True

    def publish_extension(self, extension_name: str, repo_name: str, force=False):
        if not self._repo_mgr.does_repo_exist(repo_name):
            return False

        uuid = self._get_extension_uid(extension_name)
        if not uuid:
            return False

        ext = self._ext_lookup(uuid)
        if not ext:
            logger.error(f"Extension not found. Did you forget to register the extension {extension_name} ?")

        if not self.publish_extension_interface(ext.name, repo_name, ext.uuid, force=force):
            logger.warning(f"Failed to publish extension:{ext.name}")
            return False

        variants = self.get_extension_variants(ext.name, ext.version, "default", ext.uuid)
        if variants is None:
            logger.warning(f"Failed to fetch variants for extension: {ext.name}")
            return False

        for var in variants:
            if not self.publish_extension_variant(ext.name, repo_name, var.target, ext.uuid):
                logger.warning(f"Failed to publish variant for extension: {ext.name} "
                                      f"variant: {target_to_str(var.target)}")
                return False

        return True

    def publish_extension_interface(self, name: str, repo_name, uuid=None, log_file=None, force=False):
        """Publish an extension interface to a repository

        Args:
            name (str): extension name
            uuid (str): uuid of the extension
            repo_name (str): name of the repository from which the extension has to be removed
        """
        logger.info("Publishing extension interface ...")
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(name)
            if not uuid:
                return False

        if self._ext_repo_lookup(uuid) != "default":
            logger.error(f"Extension {name} not found in default repository")
            return False

        ext = self._exts[uuid]
        result = self._repo_mgr.publish_extension_interface(ext, repo_name, force)

        # Write log file
        if log_file:
            log_content = "Extension Interface Publish \n"
            log_content += f"Extension: {name}\n"
            log_content += f"NGC Repository: {repo_name}\n"
            if result:
                log_content += f"Publish Result: Pass"
            else:
                log_content += f"Publish Result: Fail"

            with open(log_file, "wt") as f:
                f.write(log_content)

        return result

    def publish_extension_variant(self, name: str, repo_name: str, target_cfg: TargetConfig,
                                  uuid = None, log_file = None):
        """Publish an extension variant to a repository

        Args:
            name (str): extension name
            uuid (str): uuid of the extension
            repo_name (str): name of the repository from which the extension has to be removed
            target_cfg(TargetConfig): Target config
        """
        logger.info("Publishing extension variant ...")
        if uuid and not uuid_validator(uuid):
            return False
        else:
            uuid = self._get_extension_uid(name)
            if not uuid:
                return False

        ext = self._exts[uuid]
        target_cfg = self._format_target_config(target_cfg)
        ext_sub_dir_path = self._get_ext_subdir_path(uuid, ext.version, target_cfg)

        if self._ext_repo_lookup(uuid) != "default":
            logger.error(f"Extension {name} not found in default repository")
            return False

        if not path.isdir(ext_sub_dir_path):
            logger.error(f"Extension variant not found {ext_sub_dir_path}")
            return False

        result = self._repo_mgr.publish_extension_variant(ext, target_cfg, repo_name)

        # Write log file
        if log_file:
            log_content = "Extension Variant Publish \n"
            log_content += f"Extension: {name}\n"
            log_content += f"NGC Repository: {repo_name}\n"
            log_content += f"arch: {target_cfg.platform.arch}\n"
            log_content += f"os: {target_cfg.platform.os}\n"
            log_content += f"distribution: {target_cfg.platform.distribution}\n"

            if target_cfg.compute.cuda: log_content += f"cuda: {target_cfg.compute.cuda}\n"
            if target_cfg.compute.cudnn: log_content += f"cudnn: {target_cfg.compute.cudnn}\n"
            if target_cfg.compute.tensorrt: log_content += f"tensort: {target_cfg.compute.tensorrt}\n"
            if target_cfg.compute.deepstream: log_content += f"deepstream: {target_cfg.compute.deepstream}\n"
            if target_cfg.compute.triton: log_content += f"triton: {target_cfg.compute.triton}\n"
            if target_cfg.compute.vpi: log_content += f"vpi: {target_cfg.compute.vpi}\n"

            if result:
                log_content += f"Publish Result: Pass"
            else:
                log_content += f"Publish Result: Fail"

            with open(log_file, "wt") as f:
                f.write(log_content)

        return result


    def _cache_path(self):
        return self._config.get_cache_path()

    def _repo_path(self):
        return self._config.get_default_repo_path()

    def _get_ext_cache_path(self, eid: str):
        return path.join(self._cache_path(), eid)

    def _get_ext_dir_path(self, eid):
        dir_path = path.join(self._repo_path(), eid)
        return dir_path

    def _get_ext_subdir_path(self, eid, version, target: TargetConfig):
        subdir = path.join(self._repo_path(), get_ext_subdir(eid, version, target))
        return subdir

    def add_ngc_repo(self, name: str, apikey: str, org: str, team: str):
        return self._repo_mgr.add_ngc_repo(name, apikey, org, team)

    def repo_list(self):
        return self._repo_mgr.repo_list()

    def repo_info(self, name: str):
        return self._repo_mgr.repo_info(name)

    def sync_repo(self, name, on_progress):
        logger.info(f"Syncing repo {name} ...")
        if not name:
            logger.error(f"Invalid repo name: {name}")
            return False

        return self._repo_mgr.sync_repo(name, on_progress) and self._scan_cache()

    def remove_repo(self, name: str):
        return self._repo_mgr.remove_repo(name) and self._scan_cache()

    def _makedirs(self, dir_path):
        dir_path = self._cleanpath(dir_path)
        if not path.exists(dir_path):
            try:
                os.makedirs(dir_path, mode=0o777, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {dir_path}")
                logger.error(f"Exception info: {e}")
                return None
        return dir_path

    @staticmethod
    def _cleanpath(dir_path):
        return os.path.abspath(os.path.expanduser(dir_path))

    def _acquire_lock(self):
        timeout = 1
        while(timeout < 30):
            try:
                self._lock_fd = open(self._lock, "w")
                fcntl.lockf(self._lock_fd, fcntl.LOCK_EX)
                return True
            except OSError as e:
                if self._lock_fd is not None:
                    self._lock_fd.close()
                logger.error(f"Waiting for registry lock {timeout} s")
                logger.debug(f"Exception info: {e}")
                time.sleep(1)
                timeout += 1
        return False

    def _release_lock(self):
        try:
            # Closing the file descriptor releases the lock
            if self._lock_fd:
                self._lock_fd.close()
        except ValueError:
            logger.error("Failed to release the lock")


    def _format_target_config(self, target_cfg: TargetConfig):
        if not isinstance(target_cfg, TargetConfig) or \
           not isinstance(target_cfg.platform.arch, str) or \
           not isinstance(target_cfg.platform.os, str) or \
           not isinstance(target_cfg.platform.distribution, str):
              return None

        def _format_string(value):
            value = str(value).replace(" ","_").lower() if value else None
            return value

        arch = _format_string(target_cfg.platform.arch)
        opsys = _format_string(target_cfg.platform.os)
        distro = _format_string(target_cfg.platform.distribution)
        platform_cfg = PlatformConfig(arch, opsys, distro)

        cuda = _format_string(target_cfg.compute.cuda)
        cudnn = _format_string(target_cfg.compute.cudnn)
        tensorrt = _format_string(target_cfg.compute.tensorrt)
        deepstream = _format_string(target_cfg.compute.deepstream)
        triton = _format_string(target_cfg.compute.triton)
        vpi = _format_string(target_cfg.compute.vpi)
        compute_cfg = ComputeConfig(cuda, cudnn, tensorrt, deepstream, triton, vpi)

        return TargetConfig(platform_cfg, compute_cfg)

    def _read_graph_target_file(self, filepath: str):

        def _get_str_if_exist(node, key):
            if key in node and node[key]:
                value = node[key]
                return str(value).replace(" ","_").lower()
            return None

        node = YamlLoader().load_yaml(filepath)
        if not node:
            logger.error(f"Failed to read graph target file")
            return None

        if not "platform" in node:
            logger.error(f"Platform info missing in graph target file")
            return None
        os = _get_str_if_exist(node["platform"], "os")
        arch = _get_str_if_exist(node["platform"], "arch")
        distro = _get_str_if_exist(node["platform"], "distribution")

        if None in [os, arch, distro]:
              logger.error("Invalid platform info in graph target file")
              return None
        platform_cfg = PlatformConfig(arch, os, distro)

        if "compute" in node:
            cuda = _get_str_if_exist(node["compute"], "cuda")
            cudnn = _get_str_if_exist(node["compute"], "cudnn")
            tensorrt = _get_str_if_exist(node["compute"], "tensorrt")
            deepstream = _get_str_if_exist(node["compute"], "deepstream")
            triton = _get_str_if_exist(node["compute"], "triton")
            vpi = _get_str_if_exist(node["compute"], "vpi")
            compute_cfg = ComputeConfig(cuda, cudnn, tensorrt, deepstream, triton, vpi)
        else:
            compute_cfg = None

        return TargetConfig(platform_cfg, compute_cfg)

    def _get_target_config(self, target_key):
        """
        Given a target key ('x86', 'aarch'), return the TargetConfig.
        """
        self._target_configs = dict()
        # read graph targets config if exists
        nodes = YamlLoader().load_all(GRAPH_TARGETS_CONFIG)
        for node in nodes:
            _node = dict(node)
            target = _node.get('target', None)
            if not target:
                logger.error('No Target Key found! Invalid config.'
                             f'Please correct config in {GRAPH_TARGETS_CONFIG}')

            if not "platform" in _node:
                logger.error(f"Platform info missing in graph target file")
                return None
            os = dict(_node.get('platform', {})).get('os', None)
            arch = dict(_node.get('platform', {})).get('arch', None)
            distro = dict(_node.get('platform', {})).get('distribution', None)
            if None in [os, arch, distro]:
              logger.error("Invalid platform info in graph target file")
              return None
            platform_cfg = PlatformConfig(arch, os, distro)

            if "compute" in _node:
                cuda = dict(_node.get('compute', {})).get('cuda', None)
                cudnn = dict(_node.get('compute', {})).get('cudnn', None)
                tensorrt = dict(_node.get('compute', {})).get('tensorrt', None)
                deepstream = dict(_node.get('compute', {})).get('deepstream', None)
                triton = dict(_node.get('compute', {})).get('triton', None)
                vpi = dict(_node.get('compute', {})).get('vpi', None)
                compute_cfg = ComputeConfig(cuda, cudnn, tensorrt, deepstream, triton, vpi)
            else:
                compute_cfg = None

            self._target_configs[target] = TargetConfig(platform_cfg, compute_cfg)
        return self._target_configs.get(target_key, None)

    def _exts_to_map(self, exts):
        result = []
        for ext, var in exts.items():
            result.append({"extension":ext.name, "uuid": ext.uuid, "version": ext.version, "variant": target_to_str(var)})
        return result
