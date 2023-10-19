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
from gxf.std import MemoryStorageType
from gxf.std import PrimitiveType
from gxf.std import Receiver
from gxf.std import Shape
from gxf.std import TensorDescription
from gxf.std import Tensor
from gxf.std import Transmitter
from gxf.python_codelet import CodeletAdapter
import ctypes
import numpy as np
import cupy


def get_cupy_ndarray_from_tensor(tensor):
    if tensor.storage_type() != MemoryStorageType.kDevice:
        raise RuntimeError(
            "The tensor should be on device and not on host!")
    data_ptr, data_size, data_type, _, shape, strides = tensor.get_tensor_info()
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object, ctypes.c_char_p]
    unowned_mem_ptr = cupy.cuda.UnownedMemory(
        ctypes.pythonapi.PyCapsule_GetPointer(data_ptr, None), data_size, None)
    mem_ptr = cupy.cuda.MemoryPointer(unowned_mem_ptr, 0)
    cupy_array = cupy.ndarray(
        shape=shape, dtype=data_type, memptr=mem_ptr, strides=strides, order='C')
    return cupy_array

class RunFFT(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of TensorDescription.
    Receives a message on the Reciever on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(), self.cid(), self.params[f"receiver0"])
        self.tx = Transmitter.get(self.context(), self.cid(), self.params[f"transmitter0"])
        self.allocator = Allocator.get(self.context(), self.cid(), self.params[f"allocator0"])

    def tick(self):
        in0 = self.rx.receive()

        # method 1 to get all the tensors from the message.
        # In this case we only have one tensor hence tensor0
        # is the first element.
        tensor0 = Tensor.find_all_from_entity(in0)
        tensor0 = tensor0[0]

        SHAPE_SIZE=640
        SHAPE_RANK=1
        BYTES_PER_ELEMENT=8

        # The values are from CreateTensor.py, should be same as host_tensor
        assert(tensor0.get_tensor_description().shape.size()==SHAPE_SIZE)
        assert(tensor0.get_tensor_description().shape.rank()==SHAPE_RANK)
        assert(tensor0.get_tensor_description().bytes_per_element==BYTES_PER_ELEMENT)

        # if tensor0 is on device copy it to host memory for comparision
        if tensor0.storage_type() == MemoryStorageType.kDevice:
            tensor0 = get_cupy_ndarray_from_tensor(tensor0)
            tensor0 = cupy.asarray(tensor0.get())
        else:
            raise RuntimeError("data is not on device!")

        print("received tensor from src: ", tensor0[0::2] + 1j*tensor0[1::2])

        # Form complex numpy array, again assuming interleaved data
        # fft_tensor = cupy.fft.fft(tensor0[0::2] + 1j*tensor0[1::2]).astype(cupy.complex64)
        fft_tensor = cupy.fft.fft(tensor0).astype(cupy.complex64)
        print("FFT of received tensor")
        print(fft_tensor)

        cuda_msg = MessageEntity(self.context())
        cuda_tensor_description = TensorDescription(
            name="cuda_tensor",
            storage_type=MemoryStorageType.kDevice,
            shape=Shape([640]),
            element_type=PrimitiveType.kComplex64,
            bytes_per_element=8,
            strides=[8]
        )

        cuda_tensor = Tensor.add_to_entity(cuda_msg, cuda_tensor_description.name)
        cuda_tensor.reshape(cuda_tensor_description, self.allocator)

        cupy_array = get_cupy_ndarray_from_tensor(cuda_tensor)
        cupy_array[:] = fft_tensor

        self.tx.publish(cuda_msg)

        return

    def stop(self):
        pass