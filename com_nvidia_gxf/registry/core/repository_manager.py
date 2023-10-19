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


from registry.core.config import RegistryRepoConfig, NGC_NO_API_KEY, NGC_NO_ORG, NGC_NO_TEAM
from registry.core.extension import Extension
from registry.core.ngc_repository import NGCRepository, PublicNGCRepository
from registry.core.repository import Repository, LocalRepository, RemoteRepository, clean_path, RepositoryType
from registry.core.utils import PlatformConfig, TargetConfig, Variant
import registry.core.logger as log

logger = log.get_logger("Registry")
class RepositoryManager:
    """ Class handling a dictionary of repositories, and performing actions
    on the config file related to registry repositories
    """

    def __init__(self, config, database):
        self._repos = {}
        self._config = config

        for repo in self._config.get_repo_list():
            res = None
            if repo.type == RepositoryType.LOCAL.value:
                res = LocalRepository.from_config(repo, database)
            elif repo.type == RepositoryType.REMOTE.value:
                res = RemoteRepository.from_config(repo, database)
            elif repo.type == RepositoryType.NGC.value and repo.name != "ngc-public":
                res = NGCRepository.from_config(repo, config.get_cache_path(), database)
            else:
              pass

            self._repos[repo.name] = res

        self._repos["ngc-public"] = PublicNGCRepository(config.get_cache_path(), database)
        self._database = database
        self._database.update_repository_table(self.repo_list())


    def get_type(self, name):
        if name not in self._repos:
            return None
        return self._repos[name].repo_type.value

    def _pre_check_add(self, repository):
        if repository.name in self._repos:
            logger.error("Repository name already exist, please select"
                               " another name")
            return False


        if isinstance(repository, LocalRepository):
            local_repos_same = [repo for repo in self._repos.values()
                                if isinstance(repo, LocalRepository)
                                and repo.directory_path
                                == repository.directory_path]
            if local_repos_same:
                logger.error("Repository directory already exist, please"
                                   " select another directory")
                return False

        if isinstance(repository, NGCRepository):
            if not repository.team:
                repository.team = NGC_NO_TEAM
            ngc_repos_same = [repo for repo in self._repos.values()
                              if isinstance(repo, NGCRepository)
                              and repo.org == repository.org
                              and repo.team == repository.team]

            if ngc_repos_same:
                logger.error("Repository org and team combination "
                                   "already exist, please select another "
                                   "org or another team")
                return False
        return True

    def _add(self, name, repository):
        res = self._pre_check_add(repository)
        if not res:
            return res
        repo_config = repository.to_config()
        res = self._config.set_repo_config(repo_config)
        if not res:
            logger.error("Failed to set repo config")
            return False
        self._repos[name] = repository
        self._database.add_repo(repository.to_config())
        return True

    def add_local_repo(self, name: str, directory_path: str):
        directory_path = clean_path(directory_path)
        repo = LocalRepository(name, directory_path)
        return self._add(name, repo)

    def add_remote_repo(self, name: str, directory_path: str, login: str,
                        url: str, password: str):
        if not (login and password and url):
            logger.error("Asking for remote directory without providing "
                               "all credentials")
            return False
        repo = RemoteRepository(name, directory_path, login, password)
        return self._add(name, repo)

    def add_ngc_repo(self, name: str, org: str,
                     apikey: str, team: str):
        if not (apikey and org):
            logger.error("Asking for ngc directory without providing all"
                               " credentials")
            return False
        repo = NGCRepository(name, org, apikey, team, self._config.get_cache_path(),
                             self._database)
        return self._add(name, repo)

    def remove_repo(self, name):
        if name not in self._repos:
            logger.error("The selected repository '{}' does not "
                               "exist".format(name))
            return False
        if isinstance(self._repos[name], RemoteRepository):
            logger.error("Not implemented")
            return False

        self._repos.pop(name)
        return self._database.remove_repo(name) and self._config.remove_repo_config(name)

    def does_repo_exist(self, repo_name):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False
        return True

    def repo_list(self):
        res = [elm.to_config() for elm in self._repos.values()]
        return res

    def repo_info(self, name) -> RegistryRepoConfig:
        if name not in self._repos:
            logger.error("Failed to find repository {}".format(name))
            return None
        return self._repos[name].to_config()

    def sync_repo(self, repo_name, on_progress):
        cache_path = self._config.get_cache_path()

        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return None

        single_repo = self._repos[repo_name]
        return single_repo.sync(cache_path, on_progress)

    def publish_extension_interface(self, ext: Extension, repo_name, force=False):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        if self._repos[repo_name].repo_type is not RepositoryType.NGC:
            logger.error("Non NGC repository cannot be used to "
                               "publish extension interface")
            return False

        repo = self._repos[repo_name]
        return repo.publish_extension_interface(ext, self._config.get_default_repo_path(), force)

    def publish_extension_variant(self, ext: Extension, target_cfg: TargetConfig, repo_name):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        if self._repos[repo_name].repo_type is not RepositoryType.NGC:
            logger.error("Non NGC repository cannot be used to "
                               "publish extension variant")
            return False

        repo = self._repos[repo_name]
        repo_path = self._config.get_default_repo_path()
        return repo.publish_extension_variant(ext, target_cfg, repo_path)

    def import_extension_py_srcs(self, repo_name, import_path, uid:str, ext_ver: str, ext_name):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        repo = self._repos[repo_name]
        return repo.import_py_srcs(import_path, uid, ext_ver, ext_name)

    def import_extension_interface(self, repo_name, import_path, uid:str, ext_ver: str):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        repo = self._repos[repo_name]
        return repo.import_extension_interface(import_path, uid, ext_ver)

    def import_extension_variant(self, repo_name, import_path, uid: str, ext_ver: str,
                                 target_cfg: TargetConfig):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        repo = self._repos[repo_name]
        return repo.import_extension_variant(import_path, uid, ext_ver, target_cfg)

    def sync_extension(self, repo_name, eid, version):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False
        repo = self._repos[repo_name]

        if not repo.sync_extension(eid, version, self._config.get_cache_path()):
            return False

        logger.info(f"Updating database ...")
        if not self._database.update_dependencies():
            return False

        return self._database.update_frontend_flag(eid, version)

    def remove_extension_interface(self, repo_name, ext_name, uuid, version):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        repo = self._repos[repo_name]
        if not self._database.check_if_ext_exists(version, uuid, repo_name):
            logger.error(f"Did you forget to sync {repo_name} repo?")
            return False

        return repo.remove_extension_interface(ext_name, uuid, version) and \
               self._database.remove_interface_from_repo(repo_name, ext_name, uuid, version)


    def remove_extension_variant(self, repo_name, ext_name, uuid, version,
                                  target_cfg: TargetConfig):
        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        repo = self._repos[repo_name]
        if not self._database.check_if_ext_exists(version, uuid, repo_name):
            logger.error(f"Did you forget to sync {repo_name} repo?")
            return False

        return repo.remove_extension_variant(ext_name, uuid, version, target_cfg) \
                and self._database.remove_variant(uuid, version, repo_name, target_cfg)

    def refresh_extension(self, repo_name, uuid):
        cache_path = self._config.get_cache_path()

        if repo_name not in self._repos:
            logger.error(f"Repository not found {repo_name}")
            return False

        repo = self._repos[repo_name]
        return repo.refresh_extension(uuid, cache_path)
