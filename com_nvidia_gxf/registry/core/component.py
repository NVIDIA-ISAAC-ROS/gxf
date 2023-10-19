# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry Configuration File Manager
"""

import sys
from registry.bindings import pygxf

from typing import Set, Dict, List
from registry.core.utils import Component_Obj
from registry.core.parameter import Parameter
import registry.core.logger as log

logger = log.get_logger("Registry")

class Component:
    def __init__(self, node: Dict):
        self._cid: str = node['type_id']
        self._typename: str = node['typename']
        self._base_typename: str = node['base_typename']
        self._is_abstract: bool = node['is_abstract']
        self._desc: str = node['description']
        self._parameters: Dict[str: Parameter] = {}
        self._display_name = self._get_value_if_exists(node, "display_name")
        self._brief = self._get_value_if_exists(node, "brief")

        if "parameters" in node and node["parameters"] is not None:
            for prop in node["parameters"]:
                self._parameters[prop['key']] = Parameter.from_metadata(prop)

    @classmethod
    def from_metadata(cls, node: Dict):
        # Mandatory data fields for a Component
        required_keys = set({'typename', 'type_id', 'base_typename',
                             'description', 'is_abstract'})

        if not cls._validate_dict(required_keys, node):
            logger.error("Missing mandatory fields in component : " + str(node['typename']))
            return None
        else:
            return cls(node)

    @classmethod
    def from_gxf_core(cls, context, cid: str):
        node = pygxf.get_comp_info(context, cid)
        if not node:
            logger.error(f"Failed to read component info for {cid}")
            return None

        node['type_id'] = cid

        p_nodes = []
        param_keys = pygxf.get_param_list(context, cid)
        if param_keys is None:
            logger.error(f"Failed to read parameter list for component {cid}")
            return None

        for p_key in param_keys:
            param = Parameter.from_gxf_core(context, cid, p_key)
            if not param:
                return None

            p_node = param.to_metadata()
            p_nodes.append(p_node)
        node['parameters'] = p_nodes
        return cls.from_metadata(node)

    @property
    def cid(self):
        return self._cid

    @property
    def typename(self):
        return self._typename

    @property
    def base_typename(self):
        return self._base_typename

    @property
    def is_abstract(self):
        return self._is_abstract

    @property
    def desc(self):
        return self._desc

    @property
    def parameters(self):
        return self._parameters

    def add_parameter(self, param: Parameter):
        """ Add a parameter object to component

        Args:
            param (Parameter): new Parameter

        Returns:
            bool: Returns true if parameter was successfully added
        """

        if 'key' not in param or param['key'] in self._parameters:
            return False

        self._parameters[param['key']] = param
        return True

    def remove_parameter(self, key: str):
        """ Remove a parameter from component

        Args:
            key (str): key of the parameter

        Returns:
            bool: Returns true if parameter was successfully removed
        """

        try:
            del self._parameters[key]
            return True

        except KeyError:
            logger.error(
                f'Invalid key : {key} is not present in component : {self.typename}')
            return False

    def fetch_parameter_list(self):
        """ Fetch the list of parameters in component

        Returns:
            List[Parameter]: list of Parameter
        """

        result = [val for val in list(self._parameters.values())]
        return result

    def to_tuple(self):
        """ Create a named tuple object from component

        Returns:
            Component_Obj: named tuple of Component
        """

        result = Component_Obj(self._cid, self._typename, self._base_typename,
                               self._is_abstract, self._desc, self._display_name,
                               self._brief)
        return result

    def to_metadata(self):
        """ Create a yaml node object from component
        """

        c_node = {}
        c_node['typename'] = self._typename
        c_node['display_name'] = self._display_name
        c_node['type_id'] = self._cid
        c_node['base_typename'] = self._base_typename
        c_node['description'] = self._desc
        c_node['brief'] = self._brief
        c_node['is_abstract'] = self._is_abstract

        p_nodes = []
        for param in self._parameters.values():
            p_node = param.to_metadata()
            p_nodes.append(p_node)

        c_node['parameters'] = p_nodes
        return c_node

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
