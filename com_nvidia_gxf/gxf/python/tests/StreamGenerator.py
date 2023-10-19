"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from gxf.cuda import CudaStreamPool
from gxf.std import Allocator
from gxf.std import Clock
from gxf.std import Receiver
from gxf.std import Transmitter
from gxf.std import TensorDescription
from gxf.std import Tensor
from gxf.std import MemoryStorageType
from gxf.std import Shape
from gxf.std import PrimitiveType
from gxf.cuda import CudaStreamId
from gxf.python_codelet import CodeletAdapter


class StreamGenerator(CodeletAdapter):
    """ Python codelet to generate a stream of tensors on tick()

    Python implementation of StreamGenerator
    Generates a tensor using allocator on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.txs = [Transmitter.get(self.context(),\
                                    self.cid(),\
                                    self.params[f"transmitter{i}"]) \
                                    for i in range(0, 2)]
        self.allocators = [Allocator.get(self.context(),\
                                        self.cid(),\
                                        self.params[f"allocator{i}"]) \
                                        for i in range(0, 2)]
        self.cuda_stream_pool = CudaStreamPool.get(self.context(),\
                                                    self.cid(),\
                                                    self.params["cuda_stream_pool"])
        self.clock = Clock.get(self.context(), self.cid(), self.params['clock'])
        self.stream_cid, _ = self.cuda_stream_pool.allocate_stream()
        return

    def tick(self):
        dev_msg = MessageEntity(self.context())
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
            32,
            host_tensor_description.strides,
            host_tensor_description.storage_type,
            self.allocators[0])
        host_tensor.reshape(host_tensor_description, self.allocators[0])
        host_tensor2 = Tensor.add_to_entity(host_msg, host_tensor_description.name)
        host_tensor2.reshape_custom(
            host_tensor_description.shape,
            host_tensor_description.element_type,
            32,
            host_tensor_description.strides,
            host_tensor_description.storage_type,
            self.allocators[0])

        dev_tensor_description = TensorDescription(
            name="cuda_tensor",
            storage_type=MemoryStorageType.kDevice,
            shape=Shape([1024, 1]),
            element_type=PrimitiveType.kFloat32,
            bytes_per_element=32
        )
        dev_tensor = Tensor.add_to_entity(dev_msg, dev_tensor_description.name)
        dev_tensor.reshape_custom(
            dev_tensor_description.shape,
            dev_tensor_description.element_type,
            32,
            dev_tensor_description.strides,
            dev_tensor_description.storage_type,
            self.allocators[1])

        dev_tensor2 = Tensor.add_to_entity(dev_msg, dev_tensor_description.name)
        dev_tensor2.reshape_custom(
            dev_tensor_description.shape,
            dev_tensor_description.element_type,
            32,
            dev_tensor_description.strides,
            dev_tensor_description.storage_type,
            self.allocators[1])

        stream_id = CudaStreamId.add_to_entity(dev_msg, "CudaStream0")
        stream_id.stream_cid = self.stream_cid

        self.txs[0].publish(host_msg)
        self.txs[1].publish(dev_msg)

        return

    def stop(self):
        pass
