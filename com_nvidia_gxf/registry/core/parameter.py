# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry Parameter
"""

import logging
import sys

from registry.bindings import pygxf
from typing import Any, Dict, Set
from registry.core.utils import Parameter_Obj
import registry.core.logger as log

logger = log.get_logger("Registry")

class Parameter:
    def __init__(self, node: Dict):
        self._key: str = node['key']
        self._headline: str = node['headline']
        self._desc: str = node['description']
        self._flags = node['flags']
        self._parameter_type = node['gxf_parameter_type']
        self._default: Any = node['default']
        self._handle_type: str = node['handle_type']
        self._rank: int = node['rank']
        self._shape: list = node['shape']
        self._min_value: Any = node['min_value']
        self._max_value: Any = node['max_value']
        self._step_value: Any = node['step_value']

    @classmethod
    def from_metadata(cls, node: Dict):
        # Mandatory data fields for a Parameter
        required_keys = set({'gxf_parameter_type', 'key',
                             'headline', 'description', 'flags',
                             'handle_type', 'default'})
        # Optional data fields for a Parameter. Need this to be backward compatible with old Components
        optional_keys = set({'rank', 'shape'})

        optional_numeric_keys = set({'min_value', 'max_value', 'step_value'})

        if not cls._validate_dict(required_keys, optional_keys, optional_numeric_keys, node):
            logger = logging.getLogger("Registry")
            logger.error(f"Missing mandatory fields in parameter metadata for : {node['key']}")
            return None
        else:
            return cls(node)

    @classmethod
    def from_gxf_core(cls, context, cid: str, key: str):
        logger = logging.getLogger("Registry")
        param_info = pygxf.get_param_info(context, cid, key)
        if not param_info:
            logger.error(f"Failed to read parameter info for key {key} in component {cid}")
            return None

        return cls.from_metadata(param_info)

    @property
    def key(self):
        return self._key

    @property
    def headline(self):
        return self._headline

    @property
    def desc(self):
        return self._desc

    @property
    def flags(self):
        return self._flags

    @property
    def parameter_type(self):
        return self._parameter_type

    @property
    def default(self):
        return self._default

    @property
    def handle_type(self):
        return self._handle_type

    @property
    def rank(self):
        return self._rank

    @property
    def shape(self):
        return self._shape

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def step_value(self):
        return self._step_value

    def to_tuple(self):
        """ Create a named tuple from parameter

        Returns:
            Parameter_Obj: named tuple of Parameter
        """
        result = Parameter_Obj(self._key, self._headline, self._desc, str(self._flags),
                                str(self.parameter_type), self._default,
                                str(self._handle_type), self._rank, self._shape,
                                self._min_value, self._max_value, self._step_value)
        return result

    def to_metadata(self):
        """ Create a yaml node from parameter
        """
        p_node = {}
        p_node['key'] = self._key
        p_node['headline'] = self._headline
        p_node['description'] = self._desc
        p_node['gxf_parameter_type'] = self._parameter_type
        p_node['default'] = self._default
        p_node['handle_type'] = self._handle_type
        p_node['flags'] = self._flags
        p_node['rank'] = self._rank
        p_node['shape'] = self._shape
        p_node['min_value'] = self._min_value
        p_node['max_value'] = self._max_value
        p_node['step_value'] = self._step_value
        return p_node

    @staticmethod
    def _validate_dict(required_keys: Set, optional_keys: Set, optional_numeric_keys: Set, node: Dict):
        logger = logging.getLogger("Registry")
        for key in required_keys:
            if key not in node:
                logger.error(f"Missing Key : {key}")
                return False
        missing_keys = []
        for key in optional_keys:
            if key not in node:
            # These are newly added fields and not present in older versions of metadata in ngc
            # Add None values to maintain backward compatibility
                node[key] = None
                missing_keys.append(key)

        for key in optional_numeric_keys:
            if key not in node:
                node[key] = None

        return True
