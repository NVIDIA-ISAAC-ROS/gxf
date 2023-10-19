"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
from gxf.std import MemoryStorageType
from gxf.cuda import CudaStreamPool
from gxf.python_codelet import PyCodeletV0
from gxf.std import Clock
from gxf.std import Allocator
from gxf.std import Receiver
from gxf.std import Transmitter

import yaml

class CodeletAdapter:
    """ Base class that users implement for a python codelet.
        This class is the brige for C++ application and the python codelet.
        Contains helper functions to access codelet parameters.
    """

    def set_bridge(self, bridge):
        self.bridge = bridge

    def context(self):
        return self.bridge.context()

    def eid(self):
        return self.bridge.eid()

    def cid(self):
        return self.bridge.cid()

    def name(self):
        return self.bridge.name()

    def get_execution_timestamp(self):
        return self.bridge.get_execution_timestamp()

    def get_execution_time(self):
        return self.bridge.get_execution_time()

    def get_delta_time(self):
        return self.bridge.get_delta_time()

    def get_execution_count(self):
        return self.bridge.get_execution_count()

    def is_first_tick(self):
        return self.bridge.is_first_tick()

    def get_params(self):
        return yaml.safe_load(self.bridge.get_params())

    def tick(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass
