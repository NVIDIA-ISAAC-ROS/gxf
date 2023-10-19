# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" Registry Configuration File Manager
"""
import tempfile

import toml
import os, os.path as path
from os.path import expanduser, abspath
import pathlib

import registry.core.logger as log

logger = log.get_logger("Registry")
DEFAULT_ROOT_PATH = str(path.join(pathlib.Path.home(), ""))
DEFAULT_CONFIG_PATH = str(path.join(pathlib.Path.home(),".config","nvgraph",""))
DEFAULT_WORKSPACE_ROOT = str(path.join(pathlib.Path.home(),".cache","nvgraph_workspace",""))

GXF_CORE_PYBIND_PATH="/opt/nvidia/graph-composer/core"
X86_TARGET_CONFIG_FILE="/opt/nvidia/graph-composer/config/target_x86_64.yaml"
AARCH64_TARGET_CONFIG_FILE="/opt/nvidia/graph-composer/config/target_aarch64.yaml"

ENV_VAR_ROOT_PATH = "NVGRAPH_REGISTRY_ROOT"
ENV_VAR_CONFIG_PATH = "NVGRAPH_CONFIG_ROOT"
ENV_VAR_WORKSPACE_ROOT = "NVGRAPH_WORKSPACE_ROOT"

NGC_NO_API_KEY = "no-apikey"
NGC_NO_ORG = "no-org"
NGC_NO_TEAM = "no-team"


class RegistryRepoConfig:
    """ Class representing the config of a registry repository.
    """

    def __init__(self, config_dict):
        self.name = config_dict["name"]
        self.type = config_dict["type"]
        self.directory = config_dict["directory"]
        self.username = config_dict["username"]
        self.password = config_dict["password"]
        self.apikey = config_dict["apikey"]
        self.org = config_dict["org"]
        self.team = config_dict["team"]
        self.default = config_dict["default"]

class RegistryConfig:
    """ Class representing the configuration of a registry.
    """

    def __init__(self):

        self._config_path = self._get_default_path(ENV_VAR_CONFIG_PATH, DEFAULT_CONFIG_PATH)
        if not os.path.exists(self._config_path + "/registry.toml"):
            self._read_default_config()
        else:
            self._read_config()

        # Create sym link to core bindings if not present
        core_path = path.join(self.get_workspace_root(),"gxf/core")
        if not os.path.islink(core_path):
            logger.debug(f"Creating symlink to gxf core python bindings {core_path}")
            try:
                pathlib.Path(core_path).parent.mkdir(parents=True)
                os.symlink(GXF_CORE_PYBIND_PATH, core_path, target_is_directory=True)
            except Exception as e:
                logger.warning(f"Failed to create symlink in python path to {GXF_CORE_PYBIND_PATH}")
                logger.warning(e)

    def get_platform_config(self):
        if os.uname().machine == "x86_64":
            return self.get_x86_target_config()
        else:
            return self.get_aarch64_target_config()

    def get_x86_target_config(self):
        return X86_TARGET_CONFIG_FILE

    def get_aarch64_target_config(self):
        return AARCH64_TARGET_CONFIG_FILE

    def _get_default_path(self, env_var, default_path):
        result = default_path
        went_well = True
        if env_var in os.environ and os.environ[env_var]:
            tmp_path = self._clean_path(os.environ[env_var])
            try:
                temp_dir = tempfile.TemporaryDirectory(dir=tmp_path)
                if os.path.isdir(temp_dir.name):
                    result = tmp_path
                else:
                    went_well = False
                temp_dir.cleanup()
            except IOError:
                went_well = False

        if not went_well:
            logger.error(
                f"Invalid environment variable {env_var}."
                f" Please set it to a folder with read/write access."
                f" Now using {default_path}")

        return result

    def _read_default_config(self):
        self._config = {"cache": "",
                        "repositories": []}
        root = self._get_default_path(ENV_VAR_ROOT_PATH, DEFAULT_ROOT_PATH)

        self._config["workspace_root"] = self._get_default_path(ENV_VAR_WORKSPACE_ROOT, DEFAULT_WORKSPACE_ROOT)
        self._config["cache"] = self._clean_path(f"{root}/.cache/nvgraph_registry")
        self._config["repositories"].append(
                            {"name": "default",
                             "type": "local",
                             "directory": self._clean_path(f"{root}/.nvgraph_local_repo"),
                             "username": "",
                             "password": "",
                             "apikey": NGC_NO_API_KEY,
                             "org": NGC_NO_ORG,
                             "team": NGC_NO_TEAM,
                             "default": True})
        self._config["repositories"].append(
                            {"name": "ngc-public",
                             "type": "ngc",
                             "directory": "",
                             "username": "",
                             "password": "",
                             "apikey": NGC_NO_API_KEY,
                             "org": NGC_NO_ORG,
                             "team": NGC_NO_TEAM,
                             "default": False})

        return True

    def _write_config(self):
        pathlib.Path(self._config_path).mkdir(parents=True, exist_ok=True)
        file_path = self._config_path + "/registry.toml"
        try:
            with open(file_path, 'w+') as f:
                toml.dump(self._config, f)
        except IOError as e:
            logger.error("Cannot open %s for writing", file_path)
            logger.error(f"Exception info: {e}")
            return False
        return True

    def _read_config(self):
        file_path = self._config_path + "/registry.toml"
        try:
            self._config = toml.load(file_path)
        except toml.decoder.TomlDecodeError as e:
            logger.error(f"Failed to read config file {file_path}")
            logger.debug(f"{e}")
            return False

        return True

    def get_repo_list(self):
        repo_list = []
        for r in self._config["repositories"]:
            repo_list.append(RegistryRepoConfig(r))
        return repo_list

    def get_cache_path(self):
        return self._config["cache"]

    def get_repo_path(self, name):
        for r in self._config["repositories"]:
            if r["name"] == name:
                return r["directory"]

    def get_default_repo_path(self):
        return self.get_repo_path("default")

    def get_workspace_root(self):
        # Update it if not found
        workspace_root = self._get_default_path(ENV_VAR_WORKSPACE_ROOT, DEFAULT_WORKSPACE_ROOT)
        if "workspace_root" not in self._config.keys():
            self._config["workspace_root"] = workspace_root
            self._write_config()
        elif self._config["workspace_root"] != workspace_root:
            logger.info(f"Overriding default workspace_root with {workspace_root}")
            return workspace_root

        return self._config["workspace_root"]

    def set_repo_config(self, repo_config):
        self._config["repositories"].append(repo_config.__dict__)
        return self._write_config()

    def set_cache_path(self, cache_path):
        self._config["cache"] = cache_path
        return self._write_config()

    def remove_repo_config(self, name):
        for i in range(len(self._config["repositories"])):
            if self._config["repositories"][i]["name"] == name:
                del self._config["repositories"][i]
                break
        return self._write_config()

    @staticmethod
    def _clean_path(dir_path):
        return path.join(path.abspath(os.path.expanduser(dir_path)),"")
