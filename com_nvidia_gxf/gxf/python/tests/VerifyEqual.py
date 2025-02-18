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
from gxf.std import Clock
from gxf.std import MemoryStorageType
from gxf.std import Receiver
from gxf.std import Tensor
from gxf.python_codelet import CodeletAdapter
import numpy as np
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

class VerifyEqual(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of Ping Rx.
    Receives a message on the Receiver on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rxs = [Receiver.get(\
                    self.context(),\
                    self.cid(),\
                    self.params[f"receiver{i}"]) \
                    for i in range(0, 2)]
        self.clock = Clock.get(self.context(), self.cid(), self.params["clock"])
        self.count = 0

    def tick(self):
        in0 = self.rxs[0].receive()
        in1 = self.rxs[1].receive()


        # method 1 to get all the tensors from the message.
        # In this case we only have one tensor hence tensor0
        # is the first element.
        tensor0 = Tensor.find_all_from_entity(in0)
        tensor0 = tensor0[0]

        # method 2 to get one tensor from the message
        tensor1 = Tensor.get_from_entity(in1)
        assert(tensor0.shape() == tensor1.shape())
        assert(tensor0.element_type() == tensor1.element_type())

        # if tensor0 is on device copy it to host memory for comparison
        if tensor0.storage_type() == MemoryStorageType.kDevice:
            tensor0 = get_cupy_ndarray_from_tensor(tensor0)
            tensor0 = tensor0.get()
        elif tensor0.storage_type() == MemoryStorageType.kHost:
            tensor0 = np.array(tensor0)
        else:
            raise RuntimeError("data neither on host or device?!")

        # if tensor1 is on device copy it to host memory for comparison
        if tensor1.storage_type() == MemoryStorageType.kDevice:
            tensor1 = get_cupy_ndarray_from_tensor(tensor1)
            tensor1 = tensor1.get()
        elif tensor1.storage_type() == MemoryStorageType.kHost:
            tensor1 = np.array(tensor1)
        else:
            raise RuntimeError("data neither on host or device?!")

        assert(np.array_equal(tensor0, tensor1))
        return

    def stop(self):
        pass
