#! /usr/env/python
# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry Repository Manager
"""
import glob
import os
from os import path
from enum import Enum

import errno

import hashlib
from pathlib import Path
from stat import S_ISDIR, S_ISREG
import shutil
from tarfile import TarFile

from registry.core.config import RegistryRepoConfig, NGC_NO_API_KEY, NGC_NO_ORG, NGC_NO_TEAM
from registry.core.extension import Extension
from registry.core.utils import PlatformConfig, ComputeConfig, TargetConfig, Variant, get_ext_subdir
from registry.core.yaml_loader import YamlLoader
import registry.core.logger as log

logger = log.get_logger("Registry")

BUF_SIZE = 65536

class RepositoryType(Enum):
    LOCAL = "local"
    REMOTE = "remote"
    NGC = "ngc"


def get_all_files_from_remote_path(sftp, ssh_client, remotedir):
    queue = [remotedir]
    dict_all_files = {}
    while queue:
        rdir = queue.pop(0)
        if rdir[-1] != "/":
            rdir += "/"
        list_dir = None
        try:
            list_dir = sftp.listdir_attr(rdir)
        except IOError:
            return None
        for entry in list_dir:
            remotepath = rdir + entry.filename
            mode = entry.st_mode
            if S_ISDIR(mode):
                queue.append(remotepath)
            elif S_ISREG(mode):
                sha1_hash_res = get_remote_sha1_hash(ssh_client,
                                                     remotepath)
                if sha1_hash_res.is_err():
                    return sha1_hash_res
                fullpath_without_root = remotepath.replace(remotedir, "")
                dict_all_files[fullpath_without_root] = sha1_hash_res.value
    return dict_all_files


def get_remote_sha1_hash(ssh_client, remotepath) -> str:
    cmd = "sha1sum {}".format(remotepath)
    _, stdout_obj, stderr_obj = ssh_client.exec_command(cmd)
    stderr = stderr_obj.read().decode("utf-8")
    if stderr != "":
        if stderr[-1] == "\n":
            stderr = stderr[:-1]
        return None
    stdout = stdout_obj.read().decode("utf-8")
    sha1_hash = stdout.split(" ")[0]
    return sha1_hash


def clean_path(path):
    return os.path.abspath(os.path.expanduser(path))


class Repository:
    """ Class representing the content of a registry repository.
    """

    def __init__(self, name, repo_type: RepositoryType):
        self._name = name
        self._type = repo_type

    @property
    def repo_type(self):
        return self._type

    @property
    def name(self):
        return self._name

    def make_folder_local(self, folder_path):
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path, mode=0o777, exist_ok=True)
            except OSError:
                logger.error("Failed to create folder {}".format(folder_path))
                return False
        return True

    def remove_directory(self, directory):
        try:
            shutil.rmtree(directory)
        except IOError as e:
            logger.warning(f"Failed to remove {directory}")
            logger.warning(f"Exception {e}")

    def copy_files_when_needed(self, files_remote, files_cache, path_remote,
                               path_cache):

        for file_loc, hash_loc in files_remote.items():
            if file_loc in files_cache:
                if hash_loc == files_remote[file_loc]:
                    continue
            path_cache_current_file = path_cache + file_loc
            cache_folder_curr_file = Path(path_cache_current_file).parent
            res = self.make_folder_local(cache_folder_curr_file)
            if not res:
                return False
            path_local_current_file = path_remote + file_loc
            res = self.copy_single_file(path_local_current_file,
                                        path_cache_current_file)
            if not res:
                return False
        return

    def extract_gz_archive(self, gzpath, import_path, remove_src = False):
        with TarFile.open(gzpath, mode="r:gz") as tarball:
            for tarf in tarball:
                try:
                    tarball.extract(tarf.name, import_path)
                except IOError:
                    logger.warning(f"Failed to extract tarfile {tarf.name}."
                                   " Possible duplicate, removing the old file")
                    os.remove(path.join(import_path, tarf.name))
                    tarball.extract(tarf.name, import_path)

        if remove_src:
            os.remove(gzpath)

        return True

    @staticmethod
    def read_variant_config(target_file):
        node = YamlLoader().load_yaml(target_file)
        if not node:
            return None

        ######   Maintain backward compatability
        if "sha256" not in node:
            node["sha256"] = None

        compute_cfg=platform_cfg=None
        if "target" in node:
            target = node["target"]
            platform_cfg = PlatformConfig(target["arch"], target["os"], target["distribution"])
            compute_cfg = ComputeConfig(target["cuda"].replace("cuda-",""), None, None, None, None, None)

        if "platform" in node:
            platform = node["platform"]
            platform_cfg = PlatformConfig(platform["arch"], platform["os"], platform["distribution"])

        if "compute" in node:
            compute = node["compute"]
            compute_cfg = ComputeConfig(compute["cuda"], compute["cudnn"], compute["tensorrt"],
                                        compute["deepstream"], compute["triton"], compute["vpi"])

        target_cfg = TargetConfig(platform_cfg, compute_cfg)
        gxf_core_version = node["gxf_core_version"]
        registry_version = node["registry_version"]
        sha256 = node["sha256"]

        return Variant(gxf_core_version, registry_version, target_cfg, sha256)
class LocalRepository(Repository):
    def __init__(self, name, directory, database):
        super().__init__(name, RepositoryType.LOCAL)
        self._db = database
        self._directory_path = clean_path(directory)

        res = super().make_folder_local(self._directory_path)
        dir_path = self._directory_path
        if not res:
            logger.error(f"Failed to create directory {dir_path}")
        if not os.access(dir_path, os.R_OK):
            logger.error(f"Cannot read in directory {dir_path}")
            res = False
        if not os.access(dir_path, os.W_OK):
            logger.error(f"Cannot write in directory {dir_path}")
            res = False
        if not res:
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT),
                          dir_path)

    @property
    def directory_path(self):
        return self._directory_path

    @classmethod
    def from_config(cls, config, database):
        try:
            repo = LocalRepository(config.name, config.directory, database)
            return repo
        except KeyError:
            logger.error(f"{config.name} repository is missing fields "
                         f"in configuration file")
            return None

    def clean(self):
        try:
            shutil.rmtree(self._directory_path)
        except IOError:
            logger.error(f"Failed to clean {super().name} repository")
            return False
        return True

    def add_to_repo(self, file_path, in_repo_dir, dst_fname=None):
        full_path = self._directory_path + "/" + in_repo_dir + "/"
        res = super().make_folder_local(full_path)
        if dst_fname is not None:
            full_path = path.join(full_path, dst_fname)
        if not res:
            logger.error("Failed to create directory")
            return False

        return self.copy_single_file(file_path, full_path)

    def remove_from_repo(self, in_repo_dir):
        full_path = path.join(self._directory_path, in_repo_dir)
        if path.isdir(full_path):
            try:
                shutil.rmtree(full_path)
            except IOError:
                logger.error(f"Failed to remove {full_path}")
                return False
        return True

    def to_config(self):
        repo_config = {"name": super().name,
                       "type": "local",
                       "directory": self._directory_path,
                       "username": "",
                       "password": "",
                       "apikey": NGC_NO_API_KEY,
                       "org": NGC_NO_ORG,
                       "team": NGC_NO_TEAM,
                       "default": False}
        return RegistryRepoConfig(repo_config)

    def fun_get_all_files_from_path(self, path):
        file_dicts = {}
        path = clean_path(path)
        if not os.path.exists(path):
            logger.error(f"Directory path does not exist: {path}")
            return None

        for root, _, files in os.walk(path):
            for filename in files:
                fullpath = root + "/" + filename
                sha1 = hashlib.sha1()
                try:
                    with open(fullpath, "rb") as f:
                        data = f.read(BUF_SIZE)
                        while data:
                            sha1.update(data)
                            data = f.read(BUF_SIZE)
                except IOError:
                    logger.error(f"Couldn't open {fullpath} for reading")
                    return None
                fullpath = clean_path(fullpath)
                to_remove = path if path[-1] == "/" else path + "/"
                fullpath_without_root = fullpath.replace(to_remove, "")
                file_dicts[fullpath_without_root] = sha1.hexdigest()
        return file_dicts

    def copy_single_file(self, src, dst):
        src = clean_path(src)
        dst = clean_path(dst)
        if not os.path.exists(src):
            logger.error(f"File does not exist {src}")
            return False
        try:
            shutil.copy2(src, dst)
            if os.path.isdir(dst):
                dst = path.join(dst, os.path.basename(src))
            mode = 0o777
            if not dst.endswith(".so"):
                mode = 0o666
            os.chmod(path.join(dst), mode)
            return True
        except IOError:
            logger.warning(f"Failed to copy {src} to {dst}")
            return False

    def _get_manifest_files(self, extension_directory) -> list:
        all_files = [y for x in os.walk(extension_directory) for y in glob.glob(os.path.join(x[0], "*"))]
        regular_files = [file for file in all_files if os.path.isfile(file)]
        manifest_files = [elm for elm in regular_files if elm.split("/")[-1] == "target.yaml"]
        return manifest_files

    def _get_uuid_version_from_meta(self, extension_file):
        if not os.path.exists(extension_file):
            logger.error(f"Metadata not found: {extension_file}")
            return False
        node = YamlLoader().load_yaml(extension_file)
        if not node:
            logger.error(f"bad metadata file: {extension_file}")
            return False
        uuid = node["uuid"] if "uuid" in node else None
        version = node["version"] if "version" in node else None
        return uuid, version

    def sync(self, cache_path, on_progress):
        dir_path = self._directory_path

        for directory in os.listdir(dir_path):
            extn_dir = path.join(dir_path, directory)
            metadata_file = path.join(extn_dir, "extension.yaml")

            if not os.path.exists(metadata_file):
                logger.warning(f"Missing metadata file {metadata_file}")
                continue

            if not self._db.add_interface_to_repo(self._name, metadata_file):
                logger.warning(f"Failed to add {metadata_file} to cache")
                continue
            uuid, version = self._get_uuid_version_from_meta(metadata_file)
            if not (uuid and version):
                logger.error(f"Missing fields uuid/version in {metadata_file}, could not add variants")
                continue
            for variant_manifest in self._get_manifest_files(extn_dir):
                variant = self.read_variant_config(variant_manifest)
                if not self._db.add_variant(uuid, version, self.name, variant):
                    logger.warning(f"Failed to add variant: {variant_manifest} to cache")
                    continue

        return self._db.update_dependencies()

    def publish_extension_interface(self, ext: Extension, repo_path, force=False):
        logger.error("Not supported for local repository")
        return False

    def publish_variant(self, extn_name,  target_cfg: TargetConfig, repo_path):
        logger.error("Not supported for local repository")
        return False

    def import_py_srcs(self, import_path, eid, version, name):
        logger.error("Not supported for local repository")
        return False

    def import_extension_interface(self, import_path, uuid: str, version: str):
        logger.error("Importing extensions is not supported for local"
                           "repository")
        return False

    def import_extension_variant(self, import_path, uuid: str, version: str,
                         target_cfg: TargetConfig):
        logger.error("Importing extensions is not supported for local"
                           "repository")
        return False

    def sync_extension(self, uuid, version, cache_path):
        if not self._sanity_check(uuid, version):
            return False

        extn_dir = path.join(self._directory_path, uuid)
        metadata_file = path.join(extn_dir, "extension.yaml")
        if not self._db.add_interface_to_repo(self._name, metadata_file):
            logger.warning(f"Failed to add {metadata_file} to cache")
            return False

        uuid, version = self._get_uuid_version_from_meta(metadata_file)
        if not (uuid and version):
            logger.error(f"Missing fields uuid/version in {metadata_file}, could not add variants")
            return False

        for variant_manifest in self._get_manifest_files(extn_dir):
            variant = self.read_variant_config(variant_manifest)
            if not self._db.add_variant(uuid, version, self.name, variant):
                logger.warning(f"Failed to add variant: {variant_manifest} to cache")
                return False

        return True

    def refresh_extension(self, uuid, cache_path):
        # Local repositories have just a single version of the extension
        # hence sync the extension to fetch the latest version
        return self.sync_extension(uuid, None, cache_path)

    def remove_extension_interface(self, ext_name, uuid, version):
        if not self._sanity_check(uuid, version):
            return False
        try:
            dir_path = path.join(self._directory_path,uuid,"")
            shutil.rmtree(dir_path)
        except IOError:
            logger.error(f"Failed to remove {ext_name} extension from local repository")
            return False
        return True

    def remove_extension_variant(self, ext_name, uuid, version,
                                 target_cfg: TargetConfig):
        if not self._sanity_check(uuid, version):
            return False
        dir_path = path.join(self._directory_path, get_ext_subdir(uuid, version, target_cfg))
        if not path.isdir(dir_path):
            logger.error(f"Extension variant does not exist {dir_path}")
            return False
        try:
            shutil.rmtree(dir_path)
        except IOError:
            logger.error(f"Failed to remove extension variant from local repository")
            return False
        return True

    def _sanity_check(self, uuid, version = None):
        path_default_extn = path.join(self._directory_path,
                                      uuid, "extension.yaml")
        if not os.path.exists(path_default_extn):
            logger.error(f"Missing metadata file {path_default_extn}")
            return False

        if version is None:
            return True

        node = YamlLoader().load_yaml(path_default_extn)
        curr_extn = Extension.from_metadata(node)
        curr_extn_ver = curr_extn.version
        if version != curr_extn_ver:
            logger.error(f"Selected extension version invalid\n"
                               f"Available: {curr_extn_ver}\n"
                               f"Selected: {version}")
            return False
        return True

class RemoteRepository(Repository):
    def __init__(self, name, directory, username, password, database):
        super().__init__(name, RepositoryType.REMOTE)
        self._username = username
        self._password = password
        self._directory_path = directory
        self._db = database
        self._tmp_sftp_client = None
        self._tmp_ssh_client = None

    @property
    def username(self):
        return self._username

    @property
    def directory_path(self):
        return self._directory_path

    @classmethod
    def from_config(cls, config, database):
        try:
            repo = RemoteRepository(config.name, config.directory,
                                    config.username, config.password,
                                    database)
            return repo
        except KeyError:
            logger(f"{config.name} repository is missing fields"
                               f" in configuration file")
            return None

    def to_config(self):
        repo_config = {"name": super().name,
                       "type": "remote",
                       "directory": self._directory_path,
                       "username": self._username,
                       "password": self._password,
                       "apikey": NGC_NO_API_KEY,
                       "org": NGC_NO_ORG,
                       "team": NGC_NO_TEAM,
                       "default": False}
        return RegistryRepoConfig(repo_config)

    def fun_get_all_files_from_path(self, path) -> dict:
        if not self._tmp_sftp_client:
            logger.error(f"Missing sftp client")
            return None
        if not self._tmp_ssh_client:
            logger.error(f"Missing ssh client")
            return None
        return get_all_files_from_remote_path(self._tmp_sftp_client,
                                              self._tmp_ssh_client, path)

    def copy_single_file(self, src, dst):
        try:
            self._tmp_sftp_client.get(src, dst)
            return True
        except IOError:
            logger.warning(f"Failed to copy {src} to {dst}")
            return False

    def sync(self, cache_path, on_progress):
        logger.error("Not supported for remote repository")
        return False

    def publish_extension_interface(self, ext: Extension, repo_path, force=False):
        logger.error("Not supported for remote repository")
        return False

    def publish_extension_variant(self, extn_name, target_cfg: TargetConfig,
                                  repo_path):
        logger.error("Not supported for remote repository")
        return False

    def sync_extension(self, eid, version, cache_path):
        logger.error("Not supported for remote repository")
        return False

    def refresh_extension(self, uuid, cache_path):
        logger.error("Not supported for remote repository")
        return False

    def import_py_srcs(self, import_path, eid, version, name):
        logger.error("Not supported for remote repository")
        return False

    def import_extension_interface(self, import_path, uuid: str, version: str):
        logger.error("Not supported for remote repository")
        return False

    def import_extension_variant(self, import_path, uuid: str, version: str,
                         target_cfg: TargetConfig):
        logger.error("Not supported for remote repository")
        return False

    def remove_extension_interface(self, ext_name, uuid, version):
        logger.error("Not supported for remote repository")
        return False

    def remove_extension_variant(self, ext_name, uuid, version,
                                 target_cfg: TargetConfig):
        logger.error("Not supported for remote repository")
        return False

