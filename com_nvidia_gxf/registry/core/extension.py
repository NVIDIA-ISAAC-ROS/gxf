# Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry Extension
"""

from packaging import version
from typing import Set, Dict

from registry.bindings import pygxf

from registry.core.utils import Extension_Obj, uuid_validator, semantic_version_validator
from registry.core.utils import extension_name_validator, priority_validator
from registry.core.component import Component
import registry.core.logger as log

logger = log.get_logger("Registry")
class Extension:
    """ Registry Extension class responsible for managing gxf extension
        related metadata
    """
    def __init__(self, node: Dict):

        # Mandatory fields in both manifest and metadata
        self._extension_library: str = node["extension_library"]
        self._version: str = node["version"]
        self._name: str = node["name"]
        self._uuid: str = node["uuid"]
        self._labels: Set[str] = node["labels"]
        self._dependencies: list = node["dependencies"]

        # Update optional fields
        self._components: Dict[str: Component] = {}
        if "components" in node:
            for comp in node["components"]:
                self._components[comp["type_id"]] = Component.from_metadata(comp)

        self._registry_version = self._get_value_if_exists(node, "registry_version")
        self._gxf_core_version = self._get_value_if_exists(node, "gxf_core_version")
        self._desc: str = self._get_value_if_exists(node, "description")
        self._author: str = self._get_value_if_exists(node, "author")
        self._headers: list = self._get_value_if_exists(node, "headers")
        self._binaries: list = self._get_value_if_exists(node, "binaries")
        self._namespace: str = self._get_value_if_exists(node, "namespace")
        self._python_alias: str = self._get_value_if_exists(node, "python_alias")
        self._python_bindings: list = self._get_value_if_exists(node, "python_bindings")
        self._python_sources: list = self._get_value_if_exists(node, "python_sources")
        self._data: list = self._get_value_if_exists(node, "data")
        self._platform = self._get_value_if_exists(node, "platform")
        self._compute = self._get_value_if_exists(node, "compute")
        self._priority = self._get_value_if_exists(node, "priority")
        self._display_name = self._get_value_if_exists(node, "display_name")
        self._category = self._get_value_if_exists(node, "category")
        self._brief = self._get_value_if_exists(node, "brief")
        self._license_file = self._get_value_if_exists(node, "license_file")
        self._license = self._get_value_if_exists(node, "license")

    @classmethod
    def from_manifest(cls, manifest: Dict, registry_version):
        """ Constructor used for non x86_64 builds for which ext info
            cannot be extracted from shared library
        """
        # Mandatory data fields for an extension from manifest
        if not cls.validate_manifest(manifest):
            return None

        manifest["registry_version"] = registry_version
        info = cls._fill_from_manifest(manifest)
        manifest.update(info)
        return cls(manifest)

    @classmethod
    def from_metadata(cls, metadata: Dict):
        """ Constructor used for cached extension metadata files
        """
        # Mandatory data fields for an extension from metadata
        required_keys = set({"extension_library",
                             "uuid", "name", "description", "author",
                             "license", "version", "components"})

        # Deprecated fields in metadata
        if "id" in metadata:
            metadata["uuid"] = metadata["id"]

        if not cls._validate_dict(required_keys, metadata):
            logger.error(f"Missing mandatory fields in extension metadata")
            return None
        if not uuid_validator(metadata["uuid"]):
            return None
        if not extension_name_validator(metadata["name"]):
            return None
        if not semantic_version_validator(metadata["version"]):
            return None
        if "dependencies" in metadata and not Extension.validate_dependencies(metadata["dependencies"]):
            return None

        return cls(metadata)

    @classmethod
    def from_gxf_core(cls, context, eid: str, registry_version,
                      ext_lib, manifest = {}):
        """ Constructor to be used for x86_64 builds from extension library
            and extension manifest file
        """

        # Validate the manifest before querying info from gxf core
        if not cls.validate_manifest(manifest):
            return None

        try:
            node = pygxf.get_ext_info(context, eid)
            if not node:
                logger.error(f"Failed to read extension info for {eid}")
                return None

            node["gxf_core_version"] = pygxf.get_runtime_version(context)
            node["registry_version"] = registry_version
            node["uuid"] = eid
            node["extension_library"] = ext_lib.split("/")[-1]
        except UnicodeDecodeError:
            logger.error(f"Could not retrieve information")
            return None

        # Copy license file info
        node["license_file"] = manifest["license_file"]

        c_nodes = []
        cids = pygxf.get_comp_list(context, eid)
        if cids is None:
            logger.error(f"Failed to read component list from extension {eid}")
            return None

        for cid in cids:
            comp = Component.from_gxf_core(context, cid)
            if comp is None:
                return None

            c_node = comp.to_metadata()
            c_nodes.append(c_node)
        node["components"] = c_nodes

        if manifest["version"] != node["version"]:
            logger.error("Version in the manifest does not match the version in binary")
            logger.error("Manifest: {} Binary: {}".format(manifest["version"],node["version"]))
            return None

        info = cls._fill_from_manifest(manifest)
        node.update(info)
        return cls(node)

    @staticmethod
    def validate_manifest(manifest: Dict):

        # Mandatory data fields for an extension from manifest
        required_keys = set({"platform", "extension_library", "priority",
                             "uuid", "name", "license_file", "version"})
        if not Extension._validate_dict(required_keys, manifest):
            logger.debug("Missing mandatory fields in manifest")
            return False
        if not Extension._validate_platform_config(manifest["platform"]):
            logger.debug("Invalid platform configuration in manifest")
            return False
        if "compute" in manifest and not Extension._validate_compute_stack(manifest["compute"]):
            logger.debug("Invalid compute stack dependencies in manifest")
            return False
        if not uuid_validator(manifest["uuid"]):
            return False
        if not priority_validator(str(manifest["priority"])):
            return False
        if not extension_name_validator(manifest["name"]):
            return False
        if not semantic_version_validator(manifest["version"]):
            return False
        if "dependencies" in manifest and not Extension.validate_dependencies(manifest["dependencies"]):
            return False
        # TODO add validator for license types

        return True

    @staticmethod
    def _validate_platform_config(platform: Dict):

        required_keys = ["arch", "os", "distribution"]
        if not Extension._validate_dict(required_keys, platform):
            logger.error("Missing mandatory fields for platform attribute")
            return False
        if platform["arch"] not in ["x86_64", "aarch64", "aarch64_sbsa"]:
            logger.error("Platform attribute does not match requirement"
                         "\"arch\" has to be one of : \"x86_64\", \"aarch64\", \"aarch64_sbsa\"")
            return False
        if platform["os"] not in ["linux", "qnx"]:
            logger.error("Platform attribute does not match requirement"
                         "\"os\" has to be one of : \"linux\", \"qnx\"")
            return False
        if platform["distribution"] not in ["ubuntu_22.04", "qnx_sdp_7.1", "rhel9"]:
            logger.error("Platform attribute does not match the requirement. "
            + "\"distribution\" has to be one of: \"ubuntu_22.04\", \"qnx_sdp_7.1\"")
            return False
        return True

    @staticmethod
    def validate_dependencies(dependencies: Dict):
        required_keys = ["extension", "uuid", "version"]

        for dep in dependencies:
            if not Extension._validate_dict(required_keys, dep):
                return False
            if not uuid_validator(dep["uuid"]):
                return False
            if not semantic_version_validator(dep["version"]):
                return False
            if not extension_name_validator(dep["extension"]):
                return False

        return True

    @staticmethod
    def _validate_compute_stack(dependencies: Dict):

        supported_keys = ["cuda", "cudnn", "tensorrt", "deepstream", "triton", "vpi"]

        for dep, ver in dependencies.items():
            if dep not in supported_keys:
                logger.error(f"Unsupported dependency specified in manifest: {dep}")
                return False
            if ver is None: continue
            try:
                ver_parse = version.Version(str(ver))
            except version.InvalidVersion:
                logger.error(f"Invalid version specified for dependency {dep}: {ver}")
                return False
        return True

    @staticmethod
    def _fill_from_manifest(input_node):
        """ Fills in the default values for some optional info of an extension
        """
        output_node = {}

        if "platform" in input_node:
            output_node["platform"] = input_node["platform"]

        if "compute" in input_node:
            compute = input_node["compute"]
            output_node["compute"] = compute

        deps = []
        if "dependencies" in input_node:
            deps = input_node["dependencies"]
        output_node["dependencies"] = deps

        labels = []
        if "labels" in input_node:
            for label in input_node["labels"]:
                if label not in labels:
                    labels.append(label)
        output_node["labels"] = list(labels)

        headers = []
        if "headers" in input_node:
            headers = input_node["headers"]
        output_node["headers"] = headers

        if "python_alias" in input_node:
            output_node["python_alias"] = input_node["python_alias"]

        if "namespace" in input_node:
            output_node["namespace"] = input_node["namespace"]

        python_bindings = []
        if "python_bindings" in input_node:
            python_bindings = input_node["python_bindings"]
        output_node["python_bindings"] = python_bindings

        python_sources = []
        if "python_sources" in input_node:
            python_sources = input_node["python_sources"]
        output_node["python_sources"] = python_sources

        binaries = []
        if "binaries" in input_node:
            binaries = input_node["binaries"]
        output_node["binaries"] = binaries

        data = []
        if "data" in input_node:
            data = input_node["data"]
        output_node["data"] = data

        if "priority" in input_node:
            output_node["priority"] = input_node["priority"]

        return output_node

    @property
    def name(self):
        return self._name

    @property
    def extension_library(self):
        return self._extension_library

    @property
    def version(self):
        return self._version

    @property
    def core_version(self):
        return self._gxf_core_version

    @property
    def registry_version(self):
        return self._registry_version

    @property
    def author(self):
        return self._author

    @property
    def labels(self):
        return self._labels

    @property
    def license_file(self):
        return self._license_file

    @property
    def uuid(self):
        return self._uuid

    @property
    def components(self):
        return self._components

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def headers(self):
        return self._headers

    @property
    def python_alias(self):
        return self._python_alias

    @property
    def namespace(self):
        return self._namespace

    @property
    def python_bindings(self):
        return self._python_bindings

    @property
    def python_sources(self):
        return self._python_sources

    @property
    def binaries(self):
        return self._binaries

    @property
    def data(self):
        return self._data

    @property
    def description(self):
        return self._desc

    @property
    def target(self):
        return self._platform

    @property
    def arch(self):
        if self._platform is None:
            return None
        return self._platform["arch"]

    @property
    def os(self):
        if self._platform is None:
            return None
        return self._platform["os"]

    @property
    def distribution(self):
        if self._platform is None:
            return None
        return self._platform["distribution"]

    @property
    def cuda(self):
        if self._compute is None:
            return None
        return self._compute["cuda"]

    @property
    def cudnn(self):
        if self._compute is None:
            return None
        return self._compute["cudnn"]

    @property
    def tensorrt(self):
        if self._compute is None:
            return None
        return self._compute["tensorrt"]

    @property
    def deepstream(self):
        if self._compute is None:
            return None
        return self._compute["deepstream"]

    @property
    def triton(self):
        if self._compute is None:
            return None
        return self._compute["triton"]

    @property
    def vpi(self):
        if self._compute is None:
            return None
        return self._compute["vpi"]

    def add_component(self, comp: Component):
        """ Adds a component object to extension

        Args:
            param (Component): new Component

        Returns:
            bool: Returns true if component was successfully added
        """

        if "type_id" not in comp or comp["type_id"] in self._components:
            return False

        self._components[comp["type_id"]] = comp
        return True

    def remove_component(self, cid: str):
        """ Remove a component from extension

        Args:
            cid (str): [description]

        Returns:
            bool: Returns true if component was successfully removed
        """

        try:
            del self._components[cid]
            return True

        except KeyError:
            logger.error(f"Invalid cid : {cid} is not present in "
                   "extension : {self.name}")
            return False

    def fetch_component_list(self):
        """ Fetch the list of components in extension

        Returns:
            List[Component]: list of Component
        """

        result = self._components.values()
        return result

    def to_tuple(self):
        """ Create a named tuple object from extension

        Returns:
            Extension_Obj: named tuple of Extension
        """

        result = Extension_Obj(self._uuid, self._name, self._desc,
                                  self._version, self._license, self._author,
                                  self._labels, self._display_name, self._category,
                                  self._brief)
        return result

    def to_metadata(self):
        """ Create a yaml node object from component
        """

        e_node = {}
        e_node["version"] = self._version
        e_node["extension_library"] = self._extension_library
        e_node["name"] = self._name
        e_node["author"] = self._author
        e_node["description"] = self._desc
        e_node["license"] = self._license
        e_node["license_file"] = self._license_file
        e_node["uuid"] = self._uuid
        e_node["headers"] = self._headers
        e_node["dependencies"] = self._dependencies
        e_node["gxf_core_version"] = self._gxf_core_version
        e_node["registry_version"] = self._registry_version
        e_node["python_alias"] = self._python_alias
        e_node["namespace"] = self._namespace
        e_node["python_sources"] = self._python_sources
        e_node["display_name"] = self._display_name
        e_node["brief"] = self._brief
        e_node["category"] = self._category

        # Update optional fields
        if self._labels is not None:
            e_node["labels"] = self._labels

        c_nodes = []
        for comp in self._components.values():
            c_node = comp.to_metadata()
            c_nodes.append(c_node)

        e_node["components"] = c_nodes
        return e_node

    def to_target(self):
        e_node = {}

        e_node["name"] = self._name
        e_node["gxf_core_version"] = self._gxf_core_version
        e_node["registry_version"] = self._registry_version
        e_node["platform"] = self._platform
        e_node["compute"] = self._compute
        e_node["binaries"] = self._binaries
        e_node["python_bindings"] = self._python_bindings
        e_node["data"] = self._data
        e_node["priority"] = self._priority

        return e_node

    @staticmethod
    def _validate_dict(required_keys: Set, node: Dict):
        for key in required_keys:
            if key not in node:
                logger.error("Missing Key : " + str(key))
                return False
        return True

    @staticmethod
    def _get_value_if_exists(node: Dict, key):
        if key in node:
            return node[key]
        return None