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

add_to_manifest("gxf/grpc/libgxf_grpc.so")

class GrpcServer(Component):
    '''IPC Server implementation based on Grpc
    '''
    gxf_native_type: str = "nvidia::gxf::GrpcServer"

    _validation_info_parameters = {'port': {'key': 'port', 'headline': 'GRPC port for listening', 'description': 'GRPC port for listening', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_INT32', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': 50000}, 'remote_access': {'key': 'remote_access', 'headline': 'Allow access from remote clients', 'description': 'Flag for remote access', 'gxf_parameter_type': 'GXF_PARAMETER_TYPE_BOOL', 'rank': 0, 'shape': [1], 'flags': 'GXF_PARAMETER_FLAGS_NONE', 'handle_type': 'N/A', 'default': False}}

    def __init__(self, name: str= "", **params):
        Component.__init__(self, type=self.get_gxf_type(), name=name, **params)

