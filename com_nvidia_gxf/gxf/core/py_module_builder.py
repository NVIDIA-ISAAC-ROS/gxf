#! /usr/env/python
# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" PyModule Builder
"""

import argparse
import fileinput
import os
import sys

from gxf import core

class PyModuleBuilder:
    IMPORTS = """

from gxf.core import Component
from gxf.core import add_to_manifest

add_to_manifest("{}")
"""

    HEADER = """'''
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

    PY_CLASS = """
class {}(Component):
    '''{}
    '''
    gxf_native_type: str = "{}"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)
"""

    def __init__(
        self,
        workspace,
        libraries,
        workspace_root,
        codelets_only: bool = True,
    ):
        self.codelets_only = codelets_only
        self.workspace = workspace
        self.libraries = libraries
        self.workspace_root = workspace_root
        # last library should be the target lib
        self.target_library = self.libraries[-1]
        self._outfile = os.path.join(
            os.path.dirname(self.workspace + "/" + self.target_library),
                            "Components.py")

    class Extension:

        def __init__(self, info, components):
            self.info = info
            self.components = components

    class Component:

        def __init__(self, info, params):
            self.info = info
            self.params = params

        def __lt__(self, other):
            return self.info["typename"] < other.info["typename"]

    class Param:

        def __init__(self, info):
            self.info = info
            self.name = info["key"]
            self.description = info["description"]
            self.type = info["gxf_parameter_type"]
            self.handle_type = info["handle_type"]

    def remove_workspace_root(self, path):
        result = path[path.startswith(
            self.workspace_root) and len(self.workspace_root):]
        if result[0] == "/":
            result = result[1:]
        return result

    def create_header(self):
        with open(self._outfile, "a") as f:
            f.write(self.HEADER)
            f.write(self.IMPORTS.format(
                self.remove_workspace_root(self.target_library)))

    def python_class_name(self, typename):
        typename = typename.replace(' ', '')
        if typename.find("<") == -1:
            return typename.split(":")[-1]
        else:
            start = typename.find("<")
            end = typename.rfind(">")
            typename_without_template = typename[:start]+typename[end+1:]
            template = typename[start+1:end]
            return self.python_class_name(typename_without_template) + '_' + \
                    self.python_class_name(template)

    @staticmethod
    def round_nested_list(lst):
        # Function to round float value to prevent precision loss
        if isinstance(lst, (float, int)):
            return round(lst, 6)
        elif isinstance(lst, list):
            return [PyModuleBuilder.round_nested_list(item)
                    if isinstance(item, (float, int))
                    else PyModuleBuilder.round_nested_list(item)
                    for item in lst]
        return lst

    def create_component_classes(self):
        with open(self._outfile, "a") as f:
            for extension in self._extensions:
                for component in extension.components:
                    if component.info["base_typename"] and \
                            (not self.codelets_only
                                or component.info["base_typename"]
                                == "nvidia::gxf::Codelet"):
                        py_class_name = self.python_class_name(
                            component.info["typename"])
                        py_gxf_type = component.info["typename"]
                        py_description = component.info["description"]
                        py_params = {}
                        for param in component.params:
                            py_params[param.name] = param.__dict__["info"]
                            is_float_param = param.__dict__["info"]["gxf_parameter_type"] == "GXF_PARAMETER_TYPE_FLOAT64"
                            if "info" in param.__dict__ and is_float_param:
                                if "default" in param.__dict__["info"]:
                                    if param.__dict__["info"]["default"] == float('inf'):
                                        param.__dict__["info"]["default"] = '__inf_placeholder__'
                            is_float32_param = param.__dict__["info"]["gxf_parameter_type"] == "GXF_PARAMETER_TYPE_FLOAT32"
                            if "info" in param.__dict__ and is_float32_param:
                                if "default" in param.__dict__["info"]:
                                    default_value = param.__dict__["info"]["default"]
                                    param.__dict__["info"]["default"] = self.round_nested_list(default_value)
                        f.write(
                            self.PY_CLASS.format(py_class_name,
                                                 py_description,
                                                 py_gxf_type,
                                                 py_params))

    def remove_inf_placeholder(self):
        with fileinput.FileInput(self._outfile, inplace=True) as file:
            for line in file:
                # Replace '__inf_placeholder__' with 'float('inf')' without single quotes
                line = line.replace('__inf_placeholder__', 'float(\'inf\')')
                # Remove single quotes around 'float('inf')'
                line = line.replace('\'float(\'inf\')\'', 'float(\'inf\')')
                print(line, end='')

    def create_py_module(self):

        def _cleanpath(dir_path):
            return os.path.abspath(os.path.expanduser(dir_path))

        context = core.context_create()
        self._extensions = []
        core.load_extensions(
            context, [self.workspace + "/" + lib for lib in self.libraries])
        ext_list = core.get_extension_list(context)
        ext = ext_list[-1]    # last extension is the target extension
        ext_info = core.get_extension_info(context, ext)
        comps = core.get_component_list(context, ext)
        component_list = []
        for comp in comps:
            comp_info = core.get_component_info(context, comp)
            params = core.get_param_list(context, comp)
            param_list = []
            for param in params:
                param_list.append(self.Param(
                    core.get_param_info(context, comp, param)))
            component_list.append(self.Component(comp_info, param_list))
        self._extensions.append(self.Extension(
            ext_info, sorted(component_list)))
        core.context_destroy(context)
        self.create_header()
        self.create_component_classes()
        self.remove_inf_placeholder()
        pass


def main(argv):
    parser = argparse.ArgumentParser(
        description="Arguments for py_module_builder")
    parser.add_argument(
        "-w",
        "--bazel_workspace",
        dest="workspace",
        type=str,
        help="bazel workspace directory",
    )
    parser.add_argument(
        "-p",
        "--bazel_workspace_root",
        dest="workspace_root",
        type=str,
        help="workspace directory",
    )
    parser.add_argument(
        "-l",
        "--libraries",
        dest="libraries",
        nargs="+",
        help="list of libraries in loading order",
    )
    args = parser.parse_args(argv)
    try:
        py_module_builder = PyModuleBuilder(
            workspace=args.workspace,
            libraries=args.libraries,
            workspace_root=args.workspace_root,
            codelets_only=False,
        )
        py_module_builder.create_py_module()
    except OSError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
