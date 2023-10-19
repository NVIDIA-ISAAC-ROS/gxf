"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from gxf.core import MessageEntity
from gxf.std import Allocator
from gxf.std import Clock
from gxf.std import TensorDescription
from gxf.std import Tensor
from gxf.std import Transmitter
from gxf.std import MemoryStorageType
from gxf.std import Shape
from gxf.std import PrimitiveType
from gxf.python_codelet import CodeletAdapter

import numpy as np

class CreateTensor(CodeletAdapter):
    """ Python codelet to generate a stream of tensors on tick()

    Python implementation of CreateTensor
    Generates a tensor using allocator on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.tx = Transmitter.get(self.context(),\
                                    self.cid(),\
                                    self.params["transmitter"])
        self.allocator = Allocator.get(self.context(),\
                                        self.cid(),\
                                        self.params["allocator"])
        self.clock = Clock.get(self.context(), self.cid(), self.params["clock"])

        return

    def tick(self):
        host_msg = MessageEntity(self.context())
        host_tensor_description = TensorDescription(
            name="host_tensor",
            storage_type=MemoryStorageType.kHost,
            shape=Shape([1024, 1]),
            element_type=PrimitiveType.kFloat32,
            bytes_per_element=32
        )

        host_tensor = Tensor.add_to_entity(host_msg, host_tensor_description.name)
        host_tensor.reshape_custom(
            host_tensor_description.shape,
            host_tensor_description.element_type,
            host_tensor_description.bytes_per_element,
            host_tensor_description.strides,
            host_tensor_description.storage_type,
            self.allocator)
        test = Tensor.add_np_array_as_tensor_to_entity(host_msg, "shivam", np.int32([[1, 2, 4], [2, 3, 4]]), self.allocator)

        self.tx.publish(host_msg)

        return

    def stop(self):
        pass