#! /usr/env/python
# Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" NGC Repository
"""

from functools import partial
from packaging import version
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import os, os.path as path
import shutil

from registry.core.config import RegistryRepoConfig, NGC_NO_API_KEY, NGC_NO_ORG, NGC_NO_TEAM
from registry.core.extension import Extension
from registry.core.ngc_client import NGCClient, PublicNGCClient
from registry.core.repository import Repository, RepositoryType
from registry.core.utils import PlatformConfig, TargetConfig, compute_sha256, get_ext_subdir
from registry.core.yaml_loader import YamlLoader
from registry.core.version import GXF_CORE_COMPATIBLE_VERSION, REGISTRY_CORE_VERSION
import registry.core.logger as log

logger = log.get_logger("Registry")

def _get_all_versions(ngc_client: NGCClient, ext_name):
    versions = ngc_client.get_extension_versions(ext_name)
    if not versions:
        logger.warning(f"No extension versions found for {ext_name}")
        return

    base_versions = [ver for ver in versions if ver.find("-") == -1]
    variants = [ver for ver in versions if ver.find("-") != -1]
    return (ext_name, base_versions, variants)

def _sync_extension_to_path(ngc_client: NGCClient, cache_path: str, item):
    ext_name = item[0]
    ext_version = item[1]
    ext_variants = item[2]

    import_path = path.join(cache_path, ext_name, ext_version, "")
    if path.isdir(import_path):
        shutil.rmtree(import_path)

    logger.info(f"Syncing extension {ext_name} version {ext_version}")
    if not ngc_client.pull_ext_metadata(ext_name, import_path, ext_version):
        logger.warning(f"Failed to sync extension {ext_name} version {ext_version}")
        shutil.rmtree(import_path)
        return

    for var in ext_variants:
        logger.debug(f"Syncing extension variant {var}")
        variant_path = path.join(import_path, var)
        if not ngc_client.pull_ext_target(ext_name, variant_path, var):
            logger.warning(f"Failed to sync variant {var}")
            shutil.rmtree(import_path)
            return

    return (ext_name, ext_version, import_path)

class PublicNGCRepository(Repository):
    def __init__(self, cache_path, database):
        super().__init__("ngc-public", RepositoryType.NGC)
        self._db = database
        self._ngc_client = PublicNGCClient()
        self._cache = path.join(cache_path, "ngc-public")

    def to_config(self):
        repo_config = {"name": super().name,
                       "type": "ngc",
                       "directory": "",
                       "username": "",
                       "password": "",
                       "apikey": NGC_NO_API_KEY,
                       "org": NGC_NO_ORG,
                       "team": NGC_NO_TEAM,
                       "default": False}
        return RegistryRepoConfig(repo_config)

    def sync(self, cache_path, on_progress):
        self._db.remove_repo_if_exists(self._name)
        if path.isdir(self._cache):
            self.remove_directory(self._cache)

        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return None
        logger.info(f"{len(exts)} extensions found")

        tpool = ThreadPool(cpu_count())
        get_all_versions_func = partial(_get_all_versions, self._ngc_client)
        version_results = tpool.map(get_all_versions_func, exts.values())
        tpool.close()
        tpool.join()

        # Remove none values from sync failures
        version_results = list(filter(lambda x: x is not None, version_results))
        # Prepare list of ext_name / ext_version / [ext_variants]
        ext_version_list = []
        for itr in version_results:
            for ver in itr[1]:
                var_list = []
                for var in itr[2]:
                    if ver == var.split("-")[0]:
                        var_list.append(var)
                if var_list:
                    ext_version_list.append((itr[0], ver, var_list))

        tpool = ThreadPool(cpu_count())
        get_latest_versions_func = partial(_sync_extension_to_path, self._ngc_client, cache_path)
        results = tpool.imap_unordered(get_latest_versions_func, ext_version_list)
        sync_results = []
        collected = 0
        total = len(ext_version_list)
        for r in results:
            collected += 1
            if r is not None:
                sync_results.append(r)
                if callable(on_progress):
                    on_progress(
                        int(collected*99/total),
                        f"Extension {r[0]}-{r[1]} retrieved"
                    )

        tpool.close()
        tpool.join()

        # Used to lookup uuid from name
        exts_inverted = {val: key for key, val in exts.items()}

        logger.info(f"Updating database ...")
        # Keep track of failures
        has_failed = False

        # Load data into db sequentially
        for item in sync_results:
            ext_name = item[0]
            ext_version = item[1]
            import_path = item[2]
            uuid = exts_inverted[ext_name]

            if not self.sync_extension_from_path(ext_name, uuid, ext_version, import_path):
                has_failed = True

        if has_failed:
            logger.warning("Some extensions failed to sync. Check logs at /tmp/gxf_registry.log")

        return self._db.update_dependencies()

    def sync_extension_from_path(self, ext_name, uuid, ext_version, import_path):
        metadata_file = path.join(import_path, "extension.yaml")
        if not self._check_gxf_core_and_registry_compatibility(metadata_file):
            logger.debug(f"Skipping extension {ext_name} version {ext_version}")
            return False

        if not self._db.add_interface_to_repo(self._name, metadata_file):
            logger.warning(f"Failed to add {ext_name} to cache")
            return False

        variants = [ item.path for item in os.scandir(import_path) if item.is_dir() ]
        for var in variants:
            target_file = path.join(import_path, var, "target.yaml")
            target = self.read_variant_config(target_file)
            if not target:
                logger.warning(f"Failed to add variant {var} to cache")
                continue

            if not self._db.add_variant(uuid, ext_version, self._name, target):
                logger.warning(f"Failed to add variant {var} to cache")
                continue

        return True

    def publish_extension_interface(self, ext: Extension, repo_path, force=False):
        logger.error("Not supported for public ngc repository")
        return False

    def publish_extension_variant(self, ext: Extension,  target_cfg: TargetConfig,
                                  repo_path):
        logger.error(f"Not supported for public ngc repository")
        return False

    def import_py_srcs(self, import_path, eid, base_version, ext_name):
        res = self._ngc_client.pull_py_srcs(ext_name, import_path, base_version)
        if not res:
            logger.error(f"Failed to download py srcs for extension: {ext_name}:{base_version}")
            return False

        return True

    def import_extension_interface(self, import_path, eid: str, ext_ver: str):
        exts = self._db.get_extensions_from_repo(self._name)
        if exts is None:
            return False
        elif eid not in [ext.uuid for ext in exts]:
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {eid} not found in ngc repo {self.name}")
            return False

        import_path = path.join(import_path, eid)
        res = super().make_folder_local(import_path)
        if not res:
            logger.error(f"Failed to create directory {import_path}")
            return False

        ext_name = [ext.name for ext in exts if ext.uuid == eid][0]
        res = self._ngc_client.pull_ext_interface(ext_name, import_path, ext_ver)
        if not res:
            logger.error(f"Failed to download interface for extension: {ext_name}")
            return False

        return True

    def import_extension_variant(self, import_path, eid: str, ext_ver: str,
                                 target_cfg: TargetConfig):
        exts = self._db.get_extensions_from_repo(self._name)
        if exts is None:
            return False
        elif eid not in [ext.uuid for ext in exts]:
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {eid} not found in ngc repo {self.name}")
            return False

        ext_name = [ext.name for ext in exts if ext.uuid == eid][0]
        ext_cache_path = path.join(self._cache, get_ext_subdir(eid, ext_ver, target_cfg))
        ext_cache_file = path.join(ext_cache_path, f"{eid}.tar.gz")

        # Populate cache
        if not path.isfile(ext_cache_file):
            res = super().make_folder_local(ext_cache_path)
            if not res:
                logger.error(f"Failed to create directory {ext_cache_path}")
                return False

            res = self._ngc_client.pull_ext_variant(ext_name, eid, ext_cache_path, ext_ver, target_cfg)
            if not res:
                logger.error(f"Failed to download variant for extension {ext_name}")
                return False
        else:
            logger.debug(f"Using cached variant {ext_cache_file}")

        # Try again when cache is corrupt
        hash = self._db.get_variant_hash(eid, ext_ver, self._name, target_cfg)
        if hash and hash != compute_sha256(ext_cache_file):
            logger.debug(f"Corrupt cache file {ext_cache_file}")
            os.remove(ext_cache_file)
            res = self._ngc_client.pull_ext_variant(ext_name, eid, ext_cache_path, ext_ver, target_cfg)
            if not res:
                logger.error(f"Failed to download variant for extension {ext_name}")
                return False

        try:
            self.extract_gz_archive(ext_cache_file, import_path)
        except OSError as e:
            logger.error(f"Failed to copy files for extension: {eid} version: {ext_ver}")
            logger.error(e)
            return False

        return True

    def sync_extension(self, uuid, version, cache_path, ename = None):
        if not ename:
            ename = self._get_ext_name(uuid)
        if not ename:
            return False

        import_path = path.join(cache_path, uuid, "")
        logger.info(f"Syncing extension {ename} version {version}")
        if not self._ngc_client.pull_ext_metadata(ename, import_path, version):
            return False

        metadata_file = path.join(import_path, "extension.yaml")
        if not self._check_gxf_core_and_registry_compatibility(metadata_file):
            logger.warning(f"Skipping extension {ename} version {version}")
            return False

        if not self._db.add_interface_to_repo(self._name, metadata_file):
            logger.warning(f"Failed to add {ename} to cache")
            return False

        versions = self._ngc_client.get_extension_versions(ename)
        if not versions:
            logger.warning(f"No extension versions found for {ename}")
            return False

        variants = [v for v in versions if version in v and v.find("-") != -1]
        for var in variants:
            logger.debug(f"Syncing extension variant {var}")
            if not self._ngc_client.pull_ext_target(ename, import_path, var):
                logger.warning(f"Skipping NGC sync of variant {var}")
                continue

            target_file = path.join(import_path, "target.yaml")
            target = self.read_variant_config(target_file)
            if not target:
                logger.warning(f"Failed to add variant {var} to cache")
                continue

            if not self._db.add_variant(uuid, version, self._name, target):
                logger.warning(f"Failed to add variant {var} to cache")
                continue

        self.remove_directory(import_path)
        return True

    def refresh_extension(self, uuid, cache_path):
        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return False
        elif uuid not in exts.keys():
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {uuid} not found in ngc repo {self.name}")
            return False

        latest_version = self._ngc_client.get_latest_version(exts[uuid])
        if latest_version is False:
            return False

        return self.sync_extension(uuid, latest_version, cache_path, exts[uuid])

    def _get_ext_name(self, uuid):
        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return False
        elif uuid not in exts.keys():
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {uuid} not found in ngc repo {self.name}")
            return False
        return exts[uuid]

    def remove_extension_interface(self, ext_name, uuid, version):
        logger.error("Not supported for public ngc repository")
        return False

    def remove_extension_variant(self, ext_name, uuid, version,
                                 target_cfg: TargetConfig):
        logger.error("Not supported for public ngc repository")
        return False

    def _check_gxf_core_and_registry_compatibility(self, extension_metadata_path):
        if not os.path.exists(extension_metadata_path):
            logger.error(f"Metadata not found: {extension_metadata_path}")
            return False
        node = YamlLoader().load_yaml(extension_metadata_path)
        if not node:
            logger.error(f"bad metadata file: {extension_metadata_path}")
            return False
        ext_m: Extension = Extension.from_metadata(node)
        if not ext_m:
            return False
        if ext_m.registry_version:
            ext_v = version.parse(ext_m.registry_version)
            reg_c = version.parse(REGISTRY_CORE_VERSION)
            if (ext_v.major != reg_c.major):
                logger.warning("Registry major version number does not match the current registry major version number")
                logger.warning(f"Registry version: {REGISTRY_CORE_VERSION} Extension compatible registry version: {ext_m.registry_version}")
                return False
        if ext_m.core_version:
            ext_v = version.parse(ext_m.core_version)
            reg_c = version.parse(GXF_CORE_COMPATIBLE_VERSION)
            if (ext_v.major != reg_c.major) or (ext_v.minor != reg_c.minor) :
                logger.debug(f"Incompatible gxf core version")
                logger.debug(f"Registry gxf core version: {GXF_CORE_COMPATIBLE_VERSION} Extension gxf core version: {ext_m.core_version}")
                return False
        return True

class NGCRepository(Repository):
    def __init__(self, name, apikey, org, team, cache_path, database):
        super().__init__(name, RepositoryType.NGC)
        self._apikey = apikey
        self._org = org
        self._team = team
        self._cache = path.join(cache_path, f"{org}-{team}")
        self._db = database
        self._ngc_client = NGCClient(self._apikey, self._org, self._team)

    @property
    def org(self):
        return self._org

    @property
    def team(self):
        return self._team

    @team.setter
    def team(self, team):
        self._team = team

    @classmethod
    def from_config(cls, config, cache_path, database):
        try:
            repo = NGCRepository(config.name, config.apikey,
                                 config.org, config.team, cache_path,
                                 database)
            return repo
        except KeyError:
            logger.error(f"{config.name} repository is missing fields in"
                         f" configuration file")
            return None

    def to_config(self):
        repo_config = {"name": super().name,
                       "type": "ngc",
                       "directory": "",
                       "username": "",
                       "password": "",
                       "apikey": self._apikey,
                       "org": self._org,
                       "team": self._team,
                       "default": False}
        return RegistryRepoConfig(repo_config)

    def sync(self, cache_path, on_progress):
        self._db.remove_repo_if_exists(self._name)
        if path.isdir(self._cache):
            self.remove_directory(self._cache)

        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return None
        logger.info(f"{len(exts)} extensions found")

        tpool = ThreadPool(cpu_count())
        get_all_versions_func = partial(_get_all_versions, self._ngc_client)
        version_results = tpool.map(get_all_versions_func, exts.values())
        tpool.close()
        tpool.join()

        # Remove none values from sync failures
        version_results = list(filter(lambda x: x is not None, version_results))
        # Prepare list of ext_name / ext_version / [ext_variants]
        ext_version_list = []
        for itr in version_results:
            for ver in itr[1]:
                var_list = []
                for var in itr[2]:
                    if ver == var.split("-")[0]:
                        var_list.append(var)
                if var_list:
                    ext_version_list.append((itr[0], ver, var_list))

        tpool = ThreadPool(cpu_count())
        get_latest_versions_func = partial(_sync_extension_to_path, self._ngc_client, cache_path)
        results = tpool.imap_unordered(get_latest_versions_func, ext_version_list)
        sync_results = []
        collected = 0
        total = len(ext_version_list)
        for r in results:
            collected += 1
            if r is not None:
                sync_results.append(r)
                if callable(on_progress):
                    on_progress(
                        int(collected*99/total),
                        f"Extension {r[0]}-{r[1]} retrieved"
                    )
        tpool.close()
        tpool.join()

        # Used to lookup uuid from name
        exts_inverted = {val: key for key, val in exts.items()}

        logger.info(f"Updating database ...")
        # Keep track of failures
        has_failed = False

        # Load data into db sequentially
        for item in sync_results:
            ext_name = item[0]
            ext_version = item[1]
            import_path = item[2]
            uuid = exts_inverted[ext_name]

            if not self.sync_extension_from_path(ext_name, uuid, ext_version, import_path):
                has_failed = True

        if has_failed:
            logger.warning("Some extensions failed to sync. Check logs at /tmp/gxf_registry.log")

        return self._db.update_dependencies()

    def sync_extension_from_path(self, ext_name, uuid, ext_version, import_path):
        metadata_file = path.join(import_path, "extension.yaml")
        if not self._check_gxf_core_and_registry_compatibility(metadata_file):
            logger.debug(f"Skipping extension {ext_name} version {ext_version}")
            return False

        if not self._db.add_interface_to_repo(self._name, metadata_file):
            logger.warning(f"Failed to add {ext_name} to cache")
            return False

        variants = [ item.path for item in os.scandir(import_path) if item.is_dir() ]
        for var in variants:
            target_file = path.join(import_path, var, "target.yaml")
            target = self.read_variant_config(target_file)
            if not target:
                logger.warning(f"Failed to add variant {var} to cache")
                continue

            if not self._db.add_variant(uuid, ext_version, self._name, target):
                logger.warning(f"Failed to add variant {var} to cache")
                continue

        return True

    def publish_extension_interface(self, ext: Extension, repo_path, force=False):
        return self._ngc_client.publish_extension_interface(ext, repo_path, force)

    def publish_extension_variant(self, ext: Extension,  target_cfg: PlatformConfig, repo_path):
        return self._ngc_client.publish_extension_variant(ext, target_cfg, repo_path)

    def import_py_srcs(self, import_path, eid, base_version, ext_name):
        res = self._ngc_client.pull_py_srcs(ext_name, import_path, base_version)
        if not res:
            logger.error(f"Failed to download py srcs for extension: {ext_name}:{base_version}")
            return False

        return True

    def import_extension_interface(self, import_path, eid: str, ext_ver: str):
        exts = self._db.get_extensions_from_repo(self._name)
        if exts is None:
            return False
        elif eid not in [ext.uuid for ext in exts]:
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {eid} not found in ngc repo {self.name}")
            return False

        import_path = path.join(import_path, eid)
        res = super().make_folder_local(import_path)
        if not res:
            logger.error(f"Failed to create directory {import_path}")
            return False

        ext_name = [ext.name for ext in exts if ext.uuid == eid][0]
        res = self._ngc_client.pull_ext_interface(ext_name, import_path, ext_ver)
        if not res:
            logger.error(f"Failed to download interface for extension: {ext_name}")
            return False

        return True

    def import_extension_variant(self, import_path, eid: str, ext_ver: str,
                         target_cfg:TargetConfig):
        exts = self._db.get_extensions_from_repo(self._name)
        if exts is None:
            return False
        elif eid not in [ext.uuid for ext in exts]:
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {eid} not found in ngc repo {self.name}")
            return False

        ext_name = [ext.name for ext in exts if ext.uuid == eid][0]
        ext_cache_path = path.join(self._cache, get_ext_subdir(eid, ext_ver, target_cfg))
        ext_cache_file = path.join(ext_cache_path, f"{eid}.tar.gz")

        # Populate cache
        if not path.isfile(ext_cache_file):
            res = super().make_folder_local(ext_cache_path)
            if not res:
                logger.error(f"Failed to create directory {ext_cache_path}")
                return False

            res = self._ngc_client.pull_ext_variant(ext_name, eid, ext_cache_path, ext_ver, target_cfg)
            if not res:
                logger.error(f"Failed to download variant for extension {ext_name}")
                return False
        else:
            logger.debug(f"Using cached variant {ext_cache_file}")

        # Try again when cache is corrupt
        hash = self._db.get_variant_hash(eid, ext_ver, self._name, target_cfg)
        if hash and hash != compute_sha256(ext_cache_file):
            logger.debug(f"Corrupt cache file {ext_cache_file}")
            os.remove(ext_cache_file)
            res = self._ngc_client.pull_ext_variant(ext_name, eid, ext_cache_path, ext_ver, target_cfg)
            if not res:
                logger.error(f"Failed to download variant for extension {ext_name}")
                return False

        try:
            self.extract_gz_archive(ext_cache_file, import_path)
        except OSError as e:
            logger.error(f"Failed to copy files for extension: {eid} version: {ext_ver}")
            logger.error(e)
            return False

        return True

    def sync_extension(self, uuid, version, cache_path, ename = None):
        if not ename:
            ename = self._get_ext_name(uuid)
        if not ename:
            return False

        import_path = path.join(cache_path, uuid, "")
        logger.info(f"Syncing extension {ename} version {version}")
        if not self._ngc_client.pull_ext_metadata(ename, import_path, version):
            return False

        metadata_file = path.join(import_path, "extension.yaml")
        if not self._check_gxf_core_and_registry_compatibility(metadata_file):
            logger.warning(f"Skipping extension {ename} version {version}")
            return False

        if not self._db.add_interface_to_repo(self._name, metadata_file):
            logger.warning(f"Failed to add {ename} to cache")
            return False

        versions = self._ngc_client.get_extension_versions(ename)
        if not versions:
            logger.warning(f"No extension versions found for {ename}")
            return False

        variants = [v for v in versions if version in v and v.find("-") != -1]
        for var in variants:
            logger.debug(f"Syncing extension variant {var}")
            if not self._ngc_client.pull_ext_target(ename, import_path, var):
                logger.warning(f"Skipping NGC sync of variant {var}")
                continue

            target_file = path.join(import_path, "target.yaml")
            variant = self.read_variant_config(target_file)
            if not variant:
                logger.warning(f"Failed to add variant {var} to cache")
                continue

            if not self._db.add_variant(uuid, version, self._name, variant):
                logger.warning(f"Failed to add variant {var} to cache")
                continue

        self.remove_directory(import_path)
        return True

    def remove_extension_interface(self, ext_name, uuid, version):
        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return False
        elif uuid not in exts.keys():
            logger.error(f"Extension {ext_name} not found in ngc repo {self.name}")
            logger.debug(f"Extension {uuid} not found in ngc repo {self.name}")
            return False

        return self._ngc_client.remove_extension_interface(ext_name, version)

    def remove_extension_variant(self, ext_name, uuid, version,
                                 target_cfg: TargetConfig):
        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return False
        elif uuid not in exts.keys():
            logger.error(f"Extension {ext_name} not found in ngc repo {self.name}")
            logger.debug(f"Extension {uuid} not found in ngc repo {self.name}")
            return False

        return self._ngc_client.remove_extension_variant(ext_name, uuid, version, target_cfg)

    def refresh_extension(self, uuid, cache_path):
        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return False
        elif uuid not in exts.keys():
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {uuid} not found in ngc repo {self.name}")
            return False

        latest_version = self._ngc_client.get_latest_version(exts[uuid])
        if latest_version is False:
            return False

        return self.sync_extension(uuid, latest_version, cache_path, exts[uuid])

    def _get_ext_name(self, uuid):
        exts = self._ngc_client.get_extension_list()
        if exts is None:
            return False
        elif uuid not in exts.keys():
            logger.error(f"Extension not found in ngc repo {self.name}")
            logger.debug(f"Extension {uuid} not found in ngc repo {self.name}")
            return False
        return exts[uuid]

    def _check_gxf_core_and_registry_compatibility(self, extension_metadata_path):
        if not os.path.exists(extension_metadata_path):
            logger.error(f"Metadata not found: {extension_metadata_path}")
            return False
        node = YamlLoader().load_yaml(extension_metadata_path)
        if not node:
            logger.error(f"bad metadata file: {extension_metadata_path}")
            return False
        ext_m: Extension = Extension.from_metadata(node)
        if not ext_m:
            return False
        if ext_m.registry_version:
            ext_v = version.parse(ext_m.registry_version)
            reg_c = version.parse(REGISTRY_CORE_VERSION)
            if (ext_v.major != reg_c.major):
                logger.warning("Registry major version number does not match the current registry major version number")
                logger.warning(f"Registry version: {REGISTRY_CORE_VERSION} Extension compatible registry version: {ext_m.registry_version}")
                return False
        if ext_m.core_version:
            ext_v = version.parse(ext_m.core_version)
            reg_c = version.parse(GXF_CORE_COMPATIBLE_VERSION)
            if (ext_v.major != reg_c.major) or (ext_v.minor != reg_c.minor) :
                logger.debug(f"Incompatible gxf core version.")
                logger.debug(f"Registry gxf core version: {GXF_CORE_COMPATIBLE_VERSION} Extension gxf core version: {ext_m.core_version}")
                return False
        return True
