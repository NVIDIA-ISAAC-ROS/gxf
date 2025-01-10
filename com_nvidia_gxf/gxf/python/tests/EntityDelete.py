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


class EntityDelete(CodeletAdapter):
    """ Python codelet to verify the deletion of Entity

    We test that deleting a python object deletes the corresponding
    backend C++ object and releasing any resources held by the entity
    """

    def start(self):
        self.params = self.get_params()
        self.txs = [Transmitter.get(self.context(),
                                    self.cid(),
                                    self.params[f"transmitter{i}"])
                    for i in range(0, 2)]
        self.allocators = [Allocator.get(self.context(),
                                         self.cid(),
                                         self.params[f"allocator{i}"])
                           for i in range(0, 2)]
        self.cuda_stream_pool = CudaStreamPool.get(self.context(),
                                                   self.cid(),
                                                   self.params["cuda_stream_pool"])
        self.clock = Clock.get(
            self.context(), self.cid(), self.params['clock'])
        self.stream_cid, _ = self.cuda_stream_pool.allocate_stream()
        return

    def tick(self):
        dev_msg = MessageEntity(self.context())
        host_msg = MessageEntity(self.context())
        test_msg = MessageEntity(self.context())

        test_tensor_description = TensorDescription(
            name="test_tensor",
            storage_type=MemoryStorageType.kDevice,
            shape=Shape([1024, 1]),
            element_type=PrimitiveType.kFloat32,
            bytes_per_element=4
        )
        test_tensor = Tensor.add_to_entity(
            test_msg, test_tensor_description.name)
        test_tensor.reshape(test_tensor_description, self.allocators[1])

        host_tensor_description = TensorDescription(
            name="host_tensor",
            storage_type=MemoryStorageType.kHost,
            shape=Shape([1024, 1]),
            element_type=PrimitiveType.kFloat32,
            bytes_per_element=4
        )
        host_tensor = Tensor.add_to_entity(
            host_msg, host_tensor_description.name)
        host_tensor.reshape(
            host_tensor_description,
            self.allocators[0])
        host_tensor.reshape(host_tensor_description, self.allocators[0])
        host_tensor2 = Tensor.add_to_entity(
            host_msg, host_tensor_description.name)
        host_tensor2.reshape(
            host_tensor_description,
            self.allocators[0])

        dev_tensor_description = TensorDescription(
            name="cuda_tensor",
            storage_type=MemoryStorageType.kDevice,
            shape=Shape([1024, 1]),
            element_type=PrimitiveType.kFloat32,
            bytes_per_element=4
        )
        dev_tensor = Tensor.add_to_entity(dev_msg, dev_tensor_description.name)
        dev_tensor.reshape(dev_tensor_description, self.allocators[1])

        dev_tensor2 = Tensor.add_to_entity(
            dev_msg, dev_tensor_description.name)

        # this is the unit test part that should run without any error
        # the try part should fail
        # after deleting the entity it should pass
        try:
            dev_tensor2.reshape(dev_tensor_description, self.allocators[1])
            # the above should throw an error and should never reach assert(0)
            assert(0)
        except:
            del test_msg
            dev_tensor2.reshape(dev_tensor_description, self.allocators[1])

        stream_id = CudaStreamId.add_to_entity(dev_msg, "CudaStream0")
        stream_id.stream_cid = self.stream_cid

        self.txs[0].publish(host_msg)
        self.txs[1].publish(dev_msg)

        return

    def stop(self):
        pass
