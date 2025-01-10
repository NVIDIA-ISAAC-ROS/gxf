"""
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
"""
from gxf.core import MessageEntity
from gxf.std import Allocator
from gxf.std import Clock
from gxf.std import Tensor
from gxf.std import Transmitter
from gxf.python_codelet import CodeletAdapter

import numpy as np


class CreateTensor(CodeletAdapter):
    """ Python codelet to generate a stream of tensors on tick()

    Python implementation of CreateTensor
    Generates a tensor using allocator on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.tx = Transmitter.get(self.context(),
                                  self.cid(),
                                  self.params["transmitter"])
        self.allocator = Allocator.get(self.context(),
                                       self.cid(),
                                       self.params["allocator"])
        self.clock = Clock.get(
            self.context(), self.cid(), self.params["clock"])

        return

    def tick(self):
        host_msg = MessageEntity(self.context())
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "int8_tensor", np.int8([[-128, 127, -4], [2, 3, 4.3]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "int16_tensor", np.int16([[32767, -32768, -4], [2, 3, 4.3]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "int32_tensor", np.int32([[-2147483647, -2, -4], [2, 3, 4.3]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "int64_tensor", np.int64([[-9223372036854775807, 2, 4], [2, 3, 4.3]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "float32_tensor", np.float32([[1.0, 2.4, 4.33], [2, 3, 4]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "float64_tensor", np.float64([[1, 2.234234, 4], [2, 3, 4]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "uint8_tensor", np.uint8([[1, 2, 255], [2, 3.003, 4]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "uint16_tensor", np.uint16([[1, 2, 65535], [2, 3.003, 4]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "uint32_tensor", np.uint32([[1, 2, 4294967294], [2, 3.003, 4]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "uint64_tensor", np.uint64([[1, 2, 4], [2.22, 3, 18446744073709551615]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "complex64_tensor", np.complex64([[1+2j, 2, 4], [2.22, 3, 18446744073]]), self.allocator)
        Tensor.add_np_array_as_tensor_to_entity(
            host_msg, "complex128_tensor", np.complex128([[179769313+1797308j, 2, 4], [2.22, 3, 1844551615]]), self.allocator)

        self.tx.publish(host_msg)

        return

    def stop(self):
        pass
