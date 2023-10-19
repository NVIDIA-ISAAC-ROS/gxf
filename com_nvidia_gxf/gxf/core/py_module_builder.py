#! /usr/env/python
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" PyModule Builder
"""

import sys
import argparse
from gxf import core
import os

class PyModuleBuilder:

    IMPORTS = """

from gxf.core import Component
from gxf.core import add_to_manifest

add_to_manifest("{}")
"""

    HEADER = """'''
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
"""

    PY_CLASS = """
class {}(Component):
    '''{}
    '''
    gxf_native_type: str = "{}"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)\n
"""

    def __init__(self, workspace,  libraries, workspace_root, codelets_only: bool = True,):
        self.codelets_only = codelets_only
        self.workspace = workspace
        self.libraries= libraries
        self.workspace_root = workspace_root
        self.target_library = self.libraries[-1] # last library should be the target lib
        self._outfile = os.path.join(os.path.dirname(
            self.workspace + '/' + self.target_library), "Components.py")

    class Extension:
        def __init__(self, info, components):
            self.info = info
            self.components = components

    class Component:

        def __init__(self, info, params):
            self.info = info
            self.params = params

        def __lt__(self, other):
            return self.info['typename'] < other.info['typename']

    class Param:
        def __init__(self, info):
            self.info = info
            self.name = info['key']
            self.description = info['description']
            self.type = info['gxf_parameter_type']
            self.handle_type = info['handle_type']

    def remove_workspace_root(self, path):
        result = path[path.startswith(self.workspace_root) and len(self.workspace_root):]
        if result[0] == '/':
            result = result[1:]
        return result

    def create_header(self):
        with open(self._outfile, "a") as f:
            f.write(self.HEADER)
            f.write(self.IMPORTS.format(self.remove_workspace_root(self.target_library)))


    def create_component_classes(self):
        with open(self._outfile, "a") as f:
            for extension in self._extensions:
                for component in extension.components:
                    # if component.info['base_typename'] == "nvidia::gxf::Codelet":
                    if component.info['base_typename'] and (not self.codelets_only or component.info['base_typename'] == "nvidia::gxf::Codelet"):
                        py_class_name = component.info['typename'].split(
                            ':')[-1]
                        py_gxf_type = component.info['typename']
                        py_description = component.info['description']
                        py_params ={}
                        for param in component.params:
                            py_params[param.name] = param.__dict__['info']
                        f.write(self.PY_CLASS.format(py_class_name,
                                py_description,  py_gxf_type, py_params))
                        # for param in component.params:
                        #     json.dump(param, f, sort_keys=True, indent=4, separators=(',', ': '))
        return

    def create_py_module(self):
        def _cleanpath(dir_path):
            return os.path.abspath(os.path.expanduser(dir_path))
        context = core.context_create()
        self._extensions = []
        core.load_extensions(context, [self.workspace + '/'+ lib for lib in self.libraries])
        ext_list = core.get_extension_list(context)
        ext = ext_list[-1] # last extension is the target extension
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
        self._extensions.append(self.Extension(ext_info, sorted(component_list)))
        core.context_destroy(context)
        self.create_header()
        self.create_component_classes()
        pass


def main(argv):

    parser = argparse.ArgumentParser(description='Arguments for py_module_builder')
    parser.add_argument('-w', '--bazel_workspace', dest='workspace', type=str,
                        help='bazel workspace directory')
    parser.add_argument('-p', '--bazel_workspace_root', dest='workspace_root', type=str,
                        help='workspace directory')
    parser.add_argument('-l', '--libraries', dest='libraries', nargs='+',
                        help='list of libraries in loading order')
    args = parser.parse_args(argv)
    try:
        py_module_builder = PyModuleBuilder(workspace=args.workspace,
                                            libraries=args.libraries,
                                            workspace_root=args.workspace_root,
                                            codelets_only = False)
        py_module_builder.create_py_module()
    except OSError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
