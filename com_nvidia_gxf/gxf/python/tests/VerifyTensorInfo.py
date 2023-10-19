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
from gxf.std import MemoryStorageType
from gxf.std import Receiver
from gxf.std import Shape
from gxf.std import Tensor
from gxf.python_codelet import CodeletAdapter
import ctypes
import numpy as np
class VerifyTensorInfo(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of test tensor info.
    Receives a message on the Reciever on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(),\
                                self.cid(),\
                                self.params['receiver'])
        self.count = 0

    def tick(self):
        in0 = self.rx.receive()

        tensors = Tensor.find_all_from_entity(in0)
        tensor0 = tensors[0]
        tensor1 = tensors[1]

        SHAPE_SIZE=32768
        SHAPE_RANK=2
        SHAPE_DIMENSIONS=[1024,1]
        TENSOR_STRIDES=[0,0]

        # The values are from CreateTensor.py, should be same as host_tensor
        assert(tensor0.get_tensor_info()[1]==SHAPE_SIZE)
        assert(tensor0.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor0.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(tensor0.get_tensor_info()[5]==TENSOR_STRIDES)
        assert(np.array_equal(tensor1, np.int32([[1, 2, 4], [2, 3, 4]])))

        return

    def stop(self):
        pass