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
        self.tx = Transmitter.get(self.context(), self.cid(), self.params["transmitter0"])
        self.allocator = Allocator.get(self.context(), self.cid(), self.params["allocator0"])
        self.use_dlpack = bool(self.params["use_dlpack"])

        return

    def tick(self):
        # create random complex64 data
        rng = np.random.default_rng(1234)
        size = 640
        random_array = rng.standard_normal(size, dtype=np.float32)
        random_array = rng.standard_normal(size, dtype=np.float32) + 1j * rng.standard_normal(size, dtype=np.float32)
        host_msg = MessageEntity(self.context())

        if self.use_dlpack:
            # zero-copy Tensor initialization from a NumPy array
            host_tensor = Tensor.from_dlpack(random_array)
            Tensor.add_to_entity(host_msg, host_tensor, "host_tensor")
        else:
            host_tensor_description = TensorDescription(
                name="host_tensor",
                storage_type=MemoryStorageType.kHost,
                shape=Shape([random_array.size]),
                element_type=PrimitiveType.kComplex64,
                bytes_per_element=random_array.itemsize,
                strides=[random_array.strides[0]]
            )
            host_tensor = Tensor.add_to_entity(host_msg, host_tensor_description.name)
            host_tensor.reshape(host_tensor_description, self.allocator)

            numpy_array = np.array(host_tensor, copy=False)
            print(numpy_array.dtype)

            numpy_array[:] = random_array

        print("First 10 elements of transmitted tensor: ", random_array[:10])
        self.tx.publish(host_msg)
        return

    def stop(self):
        pass
