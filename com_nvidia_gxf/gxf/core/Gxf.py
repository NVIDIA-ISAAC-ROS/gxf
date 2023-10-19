'''
 SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

from .core_pybind import context_create
from .core_pybind import context_destroy
from .core_pybind import load_extensions
from .core_pybind import graph_activate
from .core_pybind import graph_run
from .core_pybind import graph_run_async
from .core_pybind import graph_wait
from .core_pybind import graph_deactivate
from .core_pybind import graph_save
from .core_pybind import graph_interrupt
from .core_pybind import gxf_set_severity
from .core_pybind import _subgraph_load_file
from .core_pybind import gxf_entity_create_info
from .core_pybind import entity_create
from .core_pybind import component_type_id
from .core_pybind import component_type_name
from .core_pybind import component_add
from .core_pybind import component_find
from .core_pybind import entity_group_create
from .core_pybind import entity_group_add
# from .core_pybind import component_add_to_interface
from .core_pybind import parameter_set_uint64
from .core_pybind import parameter_set_int32
from .core_pybind import parameter_set_int64
from .core_pybind import parameter_set_float64
from .core_pybind import parameter_set_str
from .core_pybind import parameter_set_bool
from .core_pybind import parameter_set_handle
from .core_pybind import parameter_set_1d_uint64_vector
from .core_pybind import parameter_set_1d_int32_vector
from .core_pybind import parameter_set_1d_int64_vector
from .core_pybind import parameter_set_1d_float64_vector
from .core_pybind import parameter_set_2d_uint64_vector
from .core_pybind import parameter_set_2d_int32_vector
from .core_pybind import parameter_set_2d_int64_vector
from .core_pybind import parameter_set_2d_float64_vector
from .core_pybind import parameter_set_from_yaml_node
from .core_pybind import parameter_set_path

from abc import ABC, abstractmethod
import yaml
import os
from collections.abc import Iterable
import gxf.core.logger as log
import logging


if not log.is_inited():
    log.init_logger()

logger = log.get_logger("Core")

EXTENSIONS_USED = []


def _py_log_level_to_gxf_severity(log_level):
    # GXF_SEVERITY_NONE => loggin.NOTSET
    if log_level == logging.NOTSET:
        return 0
    # GXF_SEVERITY_ERROR(1) => logging.ERROR(40)
    elif log_level == logging.ERROR:
        return 1
    # GXF_SEVERITY_WARNING(2) => logging.WARN(30)
    elif log_level == logging.WARN:
        return 2
    # GXF_SEVERITY_INFO(3) => logging.INFO(20)
    elif log_level == logging.INFO:
        return 3
    # GXF_SEVERITY_DEBUG(4) => logging.DEBUG(10)
    elif log_level == logging.DEBUG:
        return 4
    elif log_level == "VERBOSE":
        return 5
    # error
    else:
        raise RuntimeError("Incorrect log level! See logging._Level")

def add_to_manifest(extension):
    if extension not in EXTENSIONS_USED:
        EXTENSIONS_USED.append(extension)


class Node(ABC):
    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def set_params(self):
        pass


class Graph(Node):
    '''Python class wrapping the nvidia::gxf::Graph'''

    def __init__(self, name: str = '', is_static: bool = True):
        logger.setLevel(logging.INFO)
        self._context = None
        self._name = name
        self._entities = []
        self._named_entities = {}
        self._entity_groups = []
        self._named_entity_groups = {}
        self._scheduler = None
        # system entity to hold the system components such as clock, job_stats
        # entity_monitor, etc
        self._system = None
        self._is_static = is_static
        self._runtime_graph_created = False
        self._is_subgraph: bool = False
        self._parent = None
        self._subgraphs = []
        self._severity = 3
        self._ordered_nodes = []
        self._aliases = {}

    def _find_named_component(self, component: str):
        entity, component = component.split('/', 1)
        if entity not in self._named_entities:
            raise RuntimeError(f'Entity {entity} not found')
        entity = self._named_entities[entity]
        # assume that the component doesn't contain '/'
        if component not in entity.named_components:
            raise RuntimeError(
                f'Component {component} not found in {entity.name} entity')
        component = entity.named_components[component]
        return component

    @property
    def context(self):
        if self._is_subgraph:
            return self._parent.context
        else:
            if not self._context:
                self._context = context_create()
                logger.info("Context created: %s", hex(self._context))
            return self._context

    @property
    def name(self):
        return self._name

    @property
    def qualified_name(self):
        if self._is_subgraph:
            if self.parent.qualified_name:
                return self.parent.qualified_name + '-' + self._name
            else:
                return self._name
        return self._name

    def set_name(self, value):
        self._name = value

    @property
    def parent(self):
        return self._parent

    def set_parent(self, value):
        self._parent = value

    @property
    def aliases(self):
        return self._aliases

    def update_alias(self, alias, value):
        if alias not in self.aliases:
            raise ValueError("%s not set", alias)
        self.aliases[alias] = value

    def set_params(self):
        pass

    def get(self, alias):
        if alias not in self.aliases:
            raise ValueError(f"{alias} not marked visible")
        return self.aliases[alias]

    def make_visible(self, alias: str, component):
        if alias in self.aliases:
            raise RuntimeError("Duplicate: %s already exisits")
        self.aliases[alias] = component

    @property
    def is_subgraph(self):
        return self._is_subgraph

    def as_subgraph_of(self, parent):
        if self._context:
            logger.info("Adding as a subgraph. Destroying subgraph's context")
            context_destroy(self._context)
            self._context = None
        self._is_subgraph = True
        self._parent = parent

    def activate(self):
        if self._is_static and not self._runtime_graph_created:
            gxf_set_severity(self.context, self._severity)
            for node in self._ordered_nodes:
                node.activate()
                node.set_params()
            self._runtime_graph_created = True
        return

    def add(self, node):
        if isinstance(node, Entity):
            node.set_system_entity_flag(True)
            node.set_graph(self)
            self._entities.append(node)
            if node.name:
                self._named_entities[node.name] = node
                setattr(self, node.name, node)
            self._ordered_nodes.append(node)
            return node
        elif isinstance(node, EntityGroup):
            node.set_graph(self)
            self._entity_groups.append(node)
            if node.name:
                self._named_entity_groups[node.name] = node
                setattr(self, node.name, node)
            self._ordered_nodes.append(node)
            return node
        elif isinstance(node, Graph):
            logger.debug("Adding subgraph")
            if not node.name:
                logger.error("Unnamed subgraphs cannot be added!")
                raise RuntimeError("Unnamed subgraphs cannot be added!")
            node.as_subgraph_of(self)
            self._subgraphs.append(node)
            if node.name:
                self._named_entity_groups[node.name] = node
                setattr(self, node.name, node)
            self._ordered_nodes.append(node)
            return node
        else:
            raise RuntimeError(f"Cannot add {type(node)} to a Graph")

    def load_extensions(self, extension_filenames:list=None,
                        manifest_files:list=None, workspace:str=''):
        """
        Loads the GXF Extensions

        Defaults:
        * Uses env variable `GXF_WORKSPACE` as `workspace`
        * Generates `extension_filenames` based on the extensions used in the
        graph if no `extension_filenames` and `manifest_files` are passed

        Parameters:
        extension_filenames (list(string)): List of paths to the extensions
        libraries (`.so` files)
        manifest_files (list(string)): List of paths to the manifest files
        workspace (string): Path to be appended to file names in both
        `extension_filenames` and `manifest_files`
        """
        if not extension_filenames and not manifest_files:
            extension_filenames = EXTENSIONS_USED
        if not manifest_files:
            manifest_files = []
        if not workspace and os.getenv('GXF_WORKSPACE'):
            workspace = os.getenv('GXF_WORKSPACE')

        # call the backend
        load_extensions(self.context,
                        extension_filenames=extension_filenames,
                        manifest_filenames=manifest_files,
                        base_directory=workspace)
        return

    def set_severity(self, log_level=logging.INFO):
        logger.setLevel(log_level)
        self._severity = _py_log_level_to_gxf_severity(log_level)

    def run(self):
        self.activate()
        graph_activate(self.context)
        graph_run(self.context)

    def run_async(self):
        self.activate()
        graph_activate(self.context)
        graph_run_async(self.context)

    def interrupt(self):
        graph_interrupt(self.context)

    def wait(self):
        graph_wait(self.context)

    def destroy(self):
        graph_deactivate(self.context)
        context_destroy(self.context)

    def save(self, filename: str):
        self.activate()
        graph_save(self.context, filename)
        return


class Entity(Node):
    '''Entity Class wrapping the nvidia::gxf::Entity'''

    def __init__(self, name: str = "", is_system_entity: bool = False):
        self._components = []
        self._named_components = {}
        self._name = name
        self._is_system_entity = False
        self._graph: Graph = None
        # if is_system_entity:
        #     entity_info = gxf_entity_create_info(name, 1)
        # else:
        #     entity_info = gxf_entity_create_info(name, 0)
        # self._eid = entity_create(context, entity_info)

    @property
    def context(self):
        return self.graph.context

    @property
    def components(self):
        return self._components

    @property
    def named_components(self):
        return self._named_components

    @property
    def eid(self):
        return self._eid

    @eid.setter
    def eid(self, value):
        self._eid = value

    @property
    def graph(self):
        if not self._graph:
            logger.error("Entity not added to a graph")
            raise RuntimeError(
                f"Cannot perform operation before adding {self.name} to a graph!")
        return self._graph

    def set_graph(self, value):
        self._graph = value

    @property
    def name(self):
        return self._name

    @property
    def qualified_name(self):
        if self.graph.qualified_name:
            return self.graph.qualified_name + '-' + self._name
        else:
            return self._name

    @property
    def is_system_entity(self):
        return self._is_system_entity

    def set_system_entity_flag(self, is_system_entity: bool):
        self._is_system_entity = is_system_entity
        return

    def _create_entity_info(self):
        if self.is_system_entity:
            entity_info = gxf_entity_create_info(self.qualified_name, 1)
        else:
            entity_info = gxf_entity_create_info(self.qualified_name, 0)
        return entity_info

    def _is_legacy_subgraph(self, context, tid):
        return component_type_name(context, tid) == "nvidia::gxf::Subgraph"

    def _override_subgraph_params(self, component):
        if "override_params" in component.params:
            for override_component, override_params in component.params["override_params"].items():
                override_cids = component_find(
                    self.context, self.eid, component_name=override_component)
                if(len(override_cids)) > 1:
                    raise ValueError(
                        "Multiple components with the same name. Please fix the graph.")
                for param_name, param_value in override_params.items():
                    parameter_set_from_yaml_node(
                        self.context, override_cids[0], param_name, yaml.dump(yaml.safe_load(str(param_value))))

    def _load_legacy_subgraph(self, component):
        if "prerequisites" in component.params and len(component.params["prerequisites"]):
            _subgraph_load_file(self.context, component.params["location"], self.name, self.eid, str(
                component.params["prerequisites"]))
        else:
            _subgraph_load_file(
                self.context, component.params["location"], self.name, self.eid, "")
        return

    def activate(self):
        entity_info = self._create_entity_info()
        self.eid = entity_create(self.context, entity_info)
        logger.debug("Context[%s]:\tCreated %s(%d)", hex(
            self.context)[:6], self.name, self.eid)
        for component in self.components:
            component.add_to(self)
            if self._is_legacy_subgraph(self.context, component.tid):
                self._load_legacy_subgraph(component)
                self._override_subgraph_params(component)

    def set_params(self):
        for component in self.components:
            component.set_params()

    def add(self, component, visible_as: str = None):
        if (not isinstance(component, Component)):
            raise Exception("Only components can be added to entity")
        component.set_entity(self)
        self._components.append(component)
        if component.name:
            self._named_components[component.name] = component
            setattr(self, component.name, component)
        if visible_as:
            self.graph.make_visible(visible_as, component)
        return component


class EntityGroup(Node):
    '''Entity Class wrapping the nvidia::gxf::Entity'''

    def __init__(self, name: str = ""):
        self._name = name
        self._graph: Graph = None
        self._egid = None
        self._entities = []
        logger.debug("Entity Group '%s' created", name)

    @property
    def graph(self):
        if not self._graph:
            logger.error("Entity not added to a graph")
            raise RuntimeError("Cannot access graph before adding entity to one!")
        return self._graph

    def set_graph(self, value):
        self._graph = value

    @property
    def context(self):
        return self.graph.context

    @property
    def name(self):
        return self._name

    @property
    def qualified_name(self):
        if self.graph.qualified_name:
            return self.graph.qualified_name + '-' + self._name
        else:
            return self._name

    def add(self, entities):
        if isinstance(entities, Iterable):
            self._entities.extend(entities)
        else:
            self._entities.append(entities)
        return entities

    def activate(self):
        self._egid = entity_group_create(self.context, self.qualified_name)
        for entity in self._entities:
            entity_group_add(self.context, self._egid, entity.eid)
        logger.debug("Context [%s]\tEntityGroup %s(%d) created", hex(
            self.context)[:6], self.qualified_name, self._egid)

    def set_params(self):
        pass


class Component():

    gxf_native_type = "nvidia::gxf::Component"

    _validation_info_parameters = []

    def __init__(self, type: str, name: str, **params):
        '''Component Constructor. A component can only be created if it's
        being added to an entity.
        '''
        self._name: str = name
        self._type = type
        self._entity: Entity = None
        self._params: dict = params
        result = self.validate_params(params)
        if not result:
            if self._entity:
                logger.debug(self._validation_info_parameters)
                raise RuntimeError(
                    f"Parameter validation failed for {result} in {self._name} component for {self._entity._name} entity")
            else:
                logger.debug(self._validation_info_parameters)
                raise RuntimeError(
                    f"Parameter validation failed for {result} in {self._name} component")
        logger.debug("PyGxf\tComponent[%s]\tType[%s]\tCreated", name, type)

    def __repr__(self):
        if self.entity.graph.name:
            logger.debug(self.entity.qualified_name+'/'+self._name)
        return str(self.entity.qualified_name+'/'+self._name)

    @classmethod
    def get_gxf_type(cls):
        return cls.gxf_native_type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def entity(self):
        return self._entity

    @property
    def params(self):
        return self._params

    @property
    def tid(self):
        return self._tid

    @tid.setter
    def tid(self, value):
        self._tid = value

    @property
    def cid(self):
        return self._cid

    @cid.setter
    def cid(self, value):
        self._cid = value

    @property
    def context(self):
        return self.entity.context

    def add_to(self, entity):
        if not isinstance(self.type, str):
            self.tid = component_type_id(
                entity.context, self.type.get_gxf_type())
        else:
            self.tid = component_type_id(
                entity.context, self.type)
        self.cid = component_add(
            entity.context, entity.eid, self.tid, self.name)
        logger.debug("Context[%s]:\tComponent %s(%d) added to entity %s(%d)", hex(
            self.context)[:6], self.name, self.cid, entity.qualified_name, entity.eid)

    def added_to_graph(self) -> bool:
        if self.entity is None or self.entity.graph is None:
            return False
        else:
            return True

    def validate_params(self, params: dict):
        if not params or not self._validation_info_parameters:
            return True
        for param, value in params.items():
            if param not in self._validation_info_parameters:
                logger.debug(param, self._validation_info_parameters)
                return param
        return True

    def set_entity(self, entity):
        if (not self._entity):
            self._entity = entity
        else:
            raise RuntimeError("Component already added to an entity")

    def set_param(self, param, value):
        if param not in self._validation_info_parameters:
            raise ValueError("Parameter %s not found", param)
        self._params[param] = value

    def get_param(self, param):
        if param not in self._params:
            raise ValueError("Parameter %s not found", param)
        return self._params[param]

    def _set_parameter(self, name, info):
        # TODO: Bindings needed
        # if(info['dtype'] == 'uint16'):
        #     parameter_set_uint16(self._entity.context,
        #                          self._entity.eid, name, info['value'])
        # elif(info['dtype'] == 'uint32'):
        #     parameter_set_uint32(self._entity.context,
        #                          self._entity.eid, name, info['value'])
        if (info['dtype'] == 'uint64'):
            parameter_set_uint64(self._entity.context,
                                 self._entity.eid, name, info['value'])
        elif (info['dtype'] == 'int32'):
            parameter_set_int32(self._entity.context,
                                self._entity.eid, name, info['value'])
        elif (info['dtype'] == 'int64'):
            parameter_set_int64(self._entity.context,
                                self._entity.eid, name, info['value'])
        elif (info['dtype'] == 'float64'):
            parameter_set_float64(self._entity.context,
                                  self._entity.eid, name, info['value'])
        elif (info['dtype'] == 'str'):
            parameter_set_str(self._entity.context,
                              self._entity.eid, name, info['value'])
        elif (info['dtype'] == 'bool'):
            parameter_set_bool(self._entity.context,
                               self._entity.eid, name, info['value'])
        elif (info['dtype'] == 'handle'):
            parameter_set_handle(self._entity.context,
                                 self._entity.eid, name, info['value'])
        elif (info['dtype'] == '1d_uint64_vec'):
            assert 'length' in info, f"'length' needed for setting 1d vector param: {name}"
            parameter_set_1d_uint64_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['length'])
        elif (info['dtype'] == '1d_int32_vec'):
            assert 'length' in info, f"'length' needed for setting 1d vector param: {name}"
            parameter_set_1d_int32_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['length'])
        elif (info['dtype'] == '1d_int64_vec'):
            assert 'length' in info, f"'length' needed for setting 1d vector param: {name}"
            parameter_set_1d_int64_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['length'])
        elif (info['dtype'] == '1d_float64_vec'):
            assert 'length' in info, f"'length' needed for setting 1d vector param: {name}"
            parameter_set_1d_float64_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['length'])
        elif (info['dtype'] == '2d_uint64_vec'):
            assert 'height' in info and 'width' in info, f"'height' and 'width' needed for setting 2d vector param: {name}"
            parameter_set_2d_uint64_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['height'], info['width'])
        elif (info['dtype'] == '2d_int32_vec'):
            assert 'height' in info and 'width' in info, f"'height' and 'width' needed for setting 2d vector param: {name}"
            parameter_set_2d_int32_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['height'], info['width'])
        elif (info['dtype'] == '2d_int64_vec'):
            assert 'height' in info and 'width' in info, f"'height' and 'width' needed for setting 2d vector param: {name}"
            parameter_set_2d_int64_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['height'], info['width'])
        elif (info['dtype'] == '2d_float64_vec'):
            assert 'height' in info and 'width' in info, f"'height' and 'width' needed for setting 2d vector param: {name}"
            parameter_set_2d_float64_vector(
                self._entity.context, self._entity.eid, name, info['value'], info['height'], info['width'])
        return

    def set_params(self):
        if not self._params:
            return
        for param, info in self._params.items():
            # assert 'dtype' in info and 'value' in info, f"Incorrect parameter format for component: {self._name}\n\
            # Format for component params info should be {{'dtype': type, 'value': value}}"
            # self._set_parameter(param, info)
            # print("param: ", param, "info: ", info)
            if param in self._validation_info_parameters and\
                    'gxf_parameter_type' in self._validation_info_parameters[param] and\
                    self._validation_info_parameters[param]['gxf_parameter_type'] == 'GXF_PARAMETER_TYPE_FILE':
                parameter_set_path(self._entity.context,
                                   self._cid, param, str(info))
            else:
                parameter_set_from_yaml_node(
                    self._entity.context, self._cid, param, yaml.dump(yaml.safe_load(str(info))))
