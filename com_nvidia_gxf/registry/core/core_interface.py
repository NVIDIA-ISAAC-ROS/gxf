# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry
"""

from typing import Callable, List
from result import Ok, Err, Result

from registry.core.core import RegistryCore
from registry.core.utils import TargetConfig, ComputeConfig
from registry.core.utils import Component_Obj, Extension_Obj

NOT_IMPLEMENTED = Err("Not implemented!")


class Registry:
    """ Registry each method return a Result containing a string.
    """

    def __init__(self):
        """ Create an instance of core registry
        """
        self.registry_core = RegistryCore()

    def get_core_version(self) -> Result:
        """ Get registry and core versions
        """
        core_v = self.registry_core.core_version()
        if not core_v:
            return Err("Missing core version")
        return Ok(core_v)

    def get_registry_version(self) -> Result:
        """ Get registry and core versions
        """
        reg_v = self.registry_core.registry_version()
        if not reg_v:
            return Err("Missing registry version")
        return Ok(reg_v)

    def get_versions(self) -> Result:
        core_v = self.get_core_version()
        reg_v = self.get_registry_version()
        if core_v.is_ok() and reg_v.is_ok():
            return Ok((core_v, reg_v))
        return Err("Missing core or registry version")

    def set_cache_path(self, path: str) -> Result:
        """ Set registry cache path, if not set then it uses default path
        /home/<username>/.cache/nvgraph_registry
        """
        if not self.registry_core.set_cache_path(path):
            return Err("Cannot set cache path")
        return Ok("Setting new path to {}".format(path))

    def get_cache_path(self) -> Result:
        """ Get registry cache path
        """
        path = self.registry_core.get_cache_path()
        if not path:
            return Err("Cache path cannot be empty")
        return Ok(path)

    def clear_cache(self) -> Result:
        """ Clear cache content
        """
        res = self.registry_core.clear_cache()
        if not res:
            return Err("Failed to clear cache path")
        return Ok("Registry cache cleared")

    def refresh_cache(self) -> Result:
        """ Refresh cache content
        """
        res = self.registry_core.refresh_cache()
        if not res:
            return Err("Failed to refresh cache")
        return Ok("Registry cache refreshed")

    def add_ngc_repo(self, name: str, org: str, apikey: str,
                              team: str) \
            -> Result:
        """ Add NGC org repo as registry repository
        """
        if name == "default" or name == "ngc-public":
            return Err(f"Cannot create a repository named {name}")
        result = self.registry_core.add_ngc_repo(name, apikey, org, team)
        if not result:
            return Err("Failed to add ngc repo")
        return Ok("NGC repo added successfully")

    def get_repo_list(self) -> Result:
        """ List repositories
        """
        res = self.registry_core.repo_list()
        if res is None:
            return Err("Failed to get repo list")
        return Ok(res)

    def get_repo_info(self, name) -> Result:
        """ List repositories
        """
        res = self.registry_core.repo_info(name)
        if res is None:
            return Err("Failed to get repo info")
        return Ok(res)

    def sync_repo(self, name: str,
                  on_progress: Callable[[int, str], None]=None) -> Result:
        """ Rescan content in repository and update information in cache
        """
        res = self.registry_core.sync_repo(name, on_progress)
        if not res:
            return Err("Repository sync failed")
        return Ok("Repository synced")

    def remove_repo(self, name: str) -> Result:
        """ Remove repository
        """
        if name == "default" or name == "ngc-public":
            return Err(f"Cannot remove {name} repository")
        res = self.registry_core.remove_repo(name)
        if not res:
            return Err("Failed to remove repository")
        return Ok("Repo removed successfully")

    def add_extension_from_manifest(self, extn_manifest: str, meta_file: str) \
            -> Result:
        """ Add extension using manifest file, this file includes all
            information for extension such as extension file, dependencies,
             arch, platform etc.
        """
        res = self.registry_core.add_extension(extn_manifest, meta_file)
        if not res:
            return Err("Failed to load extension from manifest")
        return Ok("Loaded manifest located on: {}".format(extn_manifest))

    def remove_extension_variant(self, extn_name: str, version: str,
                                repo_name: str, target_cfg: TargetConfig,
                                uuid: str) -> Result:
        """ Remove extension variant with extension name and repo name.
        """
        result = self.registry_core.remove_extension_variant(extn_name, version, repo_name,
                                                             target_cfg, uuid)
        if not result:
            return Err("Failed to remove extension variant")
        return Ok("Extension variant removed successfully")

    def remove_extension_interface(self, extn_name: str, version: str,
                        repo_name: str, uuid: str) -> Result:
        """ Remove extension interface with extension name and repo name.
        """
        res = self.registry_core.remove_extension_interface(extn_name, version, repo_name, uuid)
        if not res:
            return Err("Failed to remove extension interface")
        return Ok("Extension interface removed successfully")

    def remove_extension(self, ext_name: str, version: str, repo_name = None):
        """Remove an extension from a repository
        """
        res = self.registry_core.remove_extension(ext_name,version,repo_name)
        if not res:
            return Err(f"Failed to remove extension {ext_name}:{version} ")
        return Ok(f"Extension {ext_name} removed successfully")

    def remove_all_extensions(self, repo_name: str):
      """ Remove all extensions from remote repository
      """
      result = self.registry_core.remove_all_extensions(repo_name)
      if not result:
          return Err("Failed to remove all extensions from remote repository")
      return Ok("Removed all extensions from remote repository")

    def get_extension_versions(self, extn_name, repo_name=None):
        res = self.registry_core.get_extension_versions(extn_name, repo_name)
        if res:
            return Ok(res)
        return Err("Failed to get extension versions")

    def get_extension_variants(self, extn_name, version):
        res = self.registry_core.get_extension_variants(extn_name, version)
        if res:
            return Ok(res)
        return Err("Failed to get extension variants")

    def get_extension_source_repo(self, extn_name, version, uuid = None):
        res = self.registry_core.get_extension_source_repo(extn_name, version, uuid)
        if res:
            return Ok(res)
        return Err("Failed to get extension source repository")

    def get_extension_dependencies(self, extn_name, version):
        res = self.registry_core.get_extension_dependencies(extn_name, version)
        if res is False:
            return Err("Failed to get extension dependencies")
        return Ok(res)

    def publish_all_extensions(self, repo_name: str, force: bool):
      """ Publish all extensions from default repository to remote repository
      """
      result = self.registry_core.publish_all_extensions(repo_name, force)
      if not result:
          return Err("Failed to publish all extensions from default repository")
      return Ok("Published all extensions from default repository")

    def publish_extension(self, extension_name: str, repo_name: str, force: bool):
      """ Publish a single extension from default repository to remote repository
      """
      result = self.registry_core.publish_extension(extension_name, repo_name, force)
      if not result:
          return Err(f"Failed to publish extension {extension_name} from default repository")
      return Ok(f"Published extension {extension_name} from default repository")

    def publish_extension_interface(self, extension_name: str, repo_name: str, uuid: str, log_file: str, force: bool) -> Result:
        """ Publish extension interface to remote repository
        """
        result = self.registry_core.publish_extension_interface(extension_name, repo_name, uuid, log_file, force)
        if not result:
            return Err("Failed to publish extension interface")
        return Ok("Extension interface published to NGC")

    def publish_extension_variant(self, extension_name: str, repo_name: str,
                          target_cfg: TargetConfig, uuid: str, log_file: str) -> Result:
        """ Publish extension variant to remote repository, it must have the interface already
        """
        result = self.registry_core.publish_extension_variant(extension_name, repo_name,
                                                      target_cfg, uuid, log_file)
        if not result:
            return Err("Failed to publish extension variant")

        return Ok("Extension variant published to NGC")

    def get_extension_list(self, repo_name=None) -> Result:
        ext_list: List[Extension_Obj] = self.registry_core \
            .fetch_extension_list(repo_name)
        if ext_list is None:
            return Err("Extension list missing or empty")
        return Ok(ext_list)

    def get_extension_uuid(self, extn_name: str) -> Result:
        res = self.registry_core.get_extension_uuid(extn_name)
        if not res:
            return Err(f"Failed to find UUID for extension {extn_name}")
        return Ok(res)

    def get_extension_info(self, tid: str) -> Result:
        """ Get extension information for specified extension UID
        """
        ext_obj: Extension_Obj = self.registry_core.fetch_extension(tid)
        if not ext_obj:
            return Err("Extension information missing")
        return Ok(ext_obj)

    def sync_extension(self, repo_name, extn_name, version, uuid):
        res = self.registry_core.sync_extension(
            repo_name, extn_name, version, uuid)
        if not res:
            return Err("Failed to synchronise extension")
        return Ok("Extension successfully synchronised")

    def set_extension_frontend_version(self, ext_name, version, uuid):
        res = self.registry_core.set_extension_frontend_version(ext_name, version, uuid)
        if not res:
            return Err("Failed to set extension frontend version")
        return Ok(f"Extension frontend version updated to {version}")

    def get_component_list(self, eid: str) -> Result:
        """ Get list of components for specified extension
        """
        component_list: List[Component_Obj] = \
            self.registry_core.fetch_component_list(eid)
        if component_list is None:
            return Err("Component list is missing or empty")
        return Ok(component_list)

    def get_component_info(self, cid: str) \
            -> Result:
        """ Get component info for specified component ID
        """
        comp: Component_Obj = self.registry_core.fetch_component(cid)
        if not comp:
            return Err("Component not found")
        return Ok(comp)

    def get_param_list(self, cid: str) -> Result:
        """ Get list of parameters using parameter type and component id
        """
        res = self.registry_core.fetch_parameter_list(cid)
        if res is None:
            return Err("Failed to fetch parameter list")
        return Ok(res)

    def install_extension(self, ext_name, target_fp=None, ext_ver = None):
        """ Installs an extension into gxf python path
        """
        res = self.registry_core.install_extension(ext_name, target_fp, ext_ver)
        if res is None:
            return Err(f"Failed to install extension {ext_name}:{ext_ver}")
        return Ok(res)

    def install_graph_with_archive(self, graphs: list,
                                   manifest_path: str,
                                   archive_path: str,
                                   in_export_dp: str,
                                   target_filepath: str) -> Result:
        """ Install graph and generate manifest file containing extensions
            required to execute the graph and a tarball containing all the
            dependencies
        """
        res = self.registry_core.install_graph_with_archive(
            graphs, manifest_path,
            archive_path, in_export_dp, target_filepath)
        if not res:
            return Err("Failed to install graph with archive")
        return Ok("Graph installed to archive")

    def install_graph_with_dir(self, graphs: list,
                                   manifest_path: str,
                                   output_path: str,
                                   in_export_dp: str,
                                   target_filepath: str) -> Result:
        """ Install graph and generate manifest file containing extensions
            required to execute the graph and a tarball containing all the
            dependencies
        """
        res = self.registry_core.install_graph_with_dir(
            graphs, manifest_path,
            output_path, in_export_dp, target_filepath)
        if not res:
            return Err("Failed to install graph with output directory")
        return Ok("Graph installed to output directory")

    def update_graph_dependencies(self, graphs: list) -> Result:
        """ Update graph to the latest version of its dependencies.
        """
        res = self.registry_core.update_graph_dependencies(graphs)
        if not res:
            return Err("Failed to update graph depdencies")
        return Ok("All graphs updated with latest depedencies!")

    def import_extension_interface(self, ext_name: str, version: str,
                                  import_dp: str, uuid: str) -> Result:
        """
        Imports an extension from an external repo(ngc) to a local directory
        """
        res = self.registry_core.import_extension_interface(ext_name, version, import_dp, uuid)
        if not res:
            return Err("Failed to import extension interface")
        return Ok("Extension interface imported successfully")

    def import_extension_variant(self, ext_name: str, version: str,
                               target_cfg: TargetConfig,
                               import_dp: str, uuid: str) -> Result:
        """
        Imports an extension from an external repo(ngc) to a local directory
        """
        res = self.registry_core.import_extension_variant(ext_name, version, target_cfg,
                                                  import_dp, uuid)
        if not res:
            return Err("Failed to import extension variant")
        return Ok("Extension variant imported successfully")

    def clean_default_repo(self) -> Result:
      """ Cleans the default repo content
      """
      res = self.registry_core.clean_default_repo()
      if not res:
          return Err("Failed to clean default repository")
      return Ok("Default repository cleaned successfully")