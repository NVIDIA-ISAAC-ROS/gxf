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
from gxf.std import MemoryStorageType
from gxf.std import Receiver
from gxf.std import Shape
from gxf.std import Tensor
from gxf.python_codelet import CodeletAdapter
import ctypes
import numpy as np

class VerifyTensors(CodeletAdapter):
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
        tensor_int8 = tensors[0]
        tensor_int16 = tensors[1]
        tensor_int32 = tensors[2]
        tensor_int64 = tensors[3]
        tensor_float32 = tensors[4]
        tensor_float64 = tensors[5]
        tensor_uint8 = tensors[6]
        tensor_uint16 = tensors[7]
        tensor_uint32 = tensors[8]
        tensor_uint64 = tensors[9]
        tensor_complex64 = tensors[10]
        tensor_complex128 = tensors[11]


        SHAPE_SIZE=6
        SHAPE_RANK=2
        SHAPE_DIMENSIONS=[2, 3]
        TENSOR_STRIDES=[3, 1]

        # The values are from CreateTensor.py, should be same as host_tensor
        assert(tensor_int8.get_tensor_info()[1]==SHAPE_SIZE)
        assert(tensor_int8.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_int8.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_int8.get_tensor_info()[5]==TENSOR_STRIDES))
        assert(np.array_equal(tensor_int8, np.int8([[-128, 127, -4], [2, 3, 4]])))

        assert(tensor_int16.get_tensor_info()[1]==SHAPE_SIZE*2)
        assert(tensor_int16.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_int16.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_int16.get_tensor_info()[5]==np.int32([2])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_int16, np.int16([[32767, -32768, -4], [2, 3, 4]])))

        assert(tensor_int32.get_tensor_info()[1]==SHAPE_SIZE*4)
        assert(tensor_int32.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_int32.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_int32.get_tensor_info()[5]==np.int32([4])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_int32, np.int32([[-2147483647, -2, -4], [2, 3, 4]])))

        assert(tensor_int64.get_tensor_info()[1]==SHAPE_SIZE*8)
        assert(tensor_int64.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_int64.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_int64.get_tensor_info()[5]==np.int32([8])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_int64, np.int64([[-9223372036854775807, 2, 4], [2, 3, 4]])))

        assert(tensor_float32.get_tensor_info()[1]==SHAPE_SIZE*4)
        assert(tensor_float32.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_float32.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_float32.get_tensor_info()[5]==np.int32([4])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_float32, np.float32([[1.0, 2.4, 4.33], [2, 3, 4]])))

        assert(tensor_float64.get_tensor_info()[1]==SHAPE_SIZE*8)
        assert(tensor_float64.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_float64.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_float64.get_tensor_info()[5]==np.int32([8])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_float64, np.float64([[1, 2.234234, 4], [2, 3, 4]])))

        assert(tensor_uint8.get_tensor_info()[1]==SHAPE_SIZE)
        assert(tensor_uint8.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_uint8.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_uint8.get_tensor_info()[5]==TENSOR_STRIDES))
        assert(np.array_equal(tensor_uint8, np.uint8([[1, 2, 255], [2, 3, 4]])))

        assert(tensor_uint16.get_tensor_info()[1]==SHAPE_SIZE*2)
        assert(tensor_uint16.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_uint16.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_uint16.get_tensor_info()[5]==np.int32([2])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_uint16, np.uint16([[1, 2, 65535], [2, 3, 4]])))

        assert(tensor_uint32.get_tensor_info()[1]==SHAPE_SIZE*4)
        assert(tensor_uint32.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_uint32.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_uint32.get_tensor_info()[5]==np.int32([4])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_uint32, np.uint32([[1, 2, 4294967294], [2, 3, 4]])))

        assert(tensor_uint64.get_tensor_info()[1]==SHAPE_SIZE*8)
        assert(tensor_uint64.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_uint64.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_uint64.get_tensor_info()[5]==np.int32([8])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_uint64, np.uint64([[1, 2, 4], [2, 3, 18446744073709551615]])))

        assert(tensor_complex64.get_tensor_info()[1]==SHAPE_SIZE*8)
        assert(tensor_complex64.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_complex64.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_complex64.get_tensor_info()[5]==np.int32([8])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_complex64, np.complex64([[1+2j, 2, 4], [2.22, 3, 18446744073]])))

        assert(tensor_complex128.get_tensor_info()[1]==SHAPE_SIZE*16)
        assert(tensor_complex128.get_tensor_info()[3]==SHAPE_RANK)
        assert(tensor_complex128.get_tensor_info()[4]==SHAPE_DIMENSIONS)
        assert(np.all(tensor_complex128.get_tensor_info()[5]==np.int32([16])*TENSOR_STRIDES))
        assert(np.array_equal(tensor_complex128, np.complex128([[179769313+1797308j, 2, 4], [2.22, 3, 1844551615]])))

        return

    def stop(self):
        pass