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
import ctypes
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

class CreateTensor(CodeletAdapter):
    """ Python codelet to generate a stream of tensors on tick()

    Python implementation of CreateTensor
    Generates a tensor using allocator on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.tx = Transmitter.get(self.context(), self.cid(), self.params[f"transmitter0"])
        self.allocator = Allocator.get(self.context(), self.cid(), self.params[f"allocator0"])
        return

    def tick(self):
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
        # Assuming interleaved data
        cupy_array[:] = cupy.random.randn(640)

        print("First 10 elements of transmitted tensor: ", cupy_array[0::2] + 1j*cupy_array[1::2])
        self.tx.publish(cuda_msg)

        return

    def stop(self):
        pass
