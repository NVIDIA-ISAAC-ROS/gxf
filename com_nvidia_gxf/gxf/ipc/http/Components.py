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

add_to_manifest("gxf/http/libgxf_http.so")

class HttpServer(Component):
    '''A light-weight http API server
    '''
    gxf_native_type: str = "nvidia::gxf::HttpServer"

    _validation_info_parameters = {'port': {'key': 'port', 'headline': 'HTTP port for listening', 'description': 'HTTP port for listening', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 8000}, 'remote_access': {'key': 'remote_access', 'headline': 'Allow access from a remote client', 'description': 'Flag to control remote access', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)


class HttpClient(Component):
    '''Interface for basic http client that works with http server inherited from IPCServer
    '''
    gxf_native_type: str = "nvidia::gxf::HttpClient"

    _validation_info_parameters = {}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)




class CppRestHttpClient(Component):
    '''A light-weight http client implementation
    '''
    gxf_native_type: str = "nvidia::gxf::CppRestHttpClient"

    _validation_info_parameters = {'server_ip_port': {'key': 'server_ip_port', 'headline': 'server ip port', 'description': 'Server IP and Port.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_STRING', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': ''}, 'use_https': {'key': 'use_https', 'headline': 'use Https', 'description': 'Use TLS(SSL). If true, protocol is https. Otherwise protocol is http.', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

