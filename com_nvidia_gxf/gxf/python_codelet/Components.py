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


from gxf.core import Component
from gxf.core import add_to_manifest

add_to_manifest("gxf/python_codelet/libgxf_python_codelet.so")

class PyCodeletV0(Component):
    '''A wrapper codelet for implementing python codelets which interfaces with CodeletAdapter
    '''
    gxf_native_type: str = "nvidia::gxf::PyCodeletV0"

    _validation_info_parameters = {'codelet_name': {'key': 'codelet_name', 'headline': 'Codelet Name', 'description': 'Name of the python codelet', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'codelet_file': {'key': 'codelet_file', 'headline': 'Absolute Codelet File Path', 'description': 'Absolute path to the file containing the codelet implementation', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 'N/A'}, 'codelet_params': {'key': 'codelet_params', 'headline': 'Params', 'description': 'Codelet params', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_OPTIONAL', 'handle_type': 'N/A', 'default': 'N/A'}}

    def __init__(self, name, codelet, **params):
        backend_params = {
            'codelet_name': codelet.__name__,
            'codelet_file': "None",
            'codelet_params': params,
        }
        Component.__init__(self, type=self.get_gxf_type(), name=name, **backend_params)

