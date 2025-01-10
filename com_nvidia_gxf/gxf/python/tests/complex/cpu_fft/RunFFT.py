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
from gxf.std import Receiver
from gxf.std import Transmitter
from gxf.std import Tensor
from gxf.python_codelet import CodeletAdapter
import numpy as np


class RunFFT(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of TensorDescription.
    Receives a message on the Receiver on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(), self.cid(), self.params["receiver0"])
        self.tx = Transmitter.get(self.context(), self.cid(), self.params["transmitter0"])
        self.allocator = Allocator.get(self.context(), self.cid(), self.params["allocator0"])
        self.use_dlpack = bool(self.params["use_dlpack"])

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
        if self.use_dlpack:
            numpy_array = np.from_dlpack(tensor0)
        else:
            numpy_array = np.array(tensor0)
        print("First 10 elements of received tensor: ", numpy_array[:10])

        # Perform FFT on the numpy tensor
        fft_tensor = np.fft.fft(numpy_array).astype(np.complex64)
        print("First 10 elements of received tensor of FFT output:", fft_tensor[:10])

        host_msg = MessageEntity(self.context())
        if self.use_dlpack:
            Tensor.add_to_entity(host_msg, Tensor.from_dlpack(fft_tensor), "host_tensor"            )
        else:
            Tensor.add_np_array_as_tensor_to_entity(
                host_msg, "host_tensor", fft_tensor, self.allocator
            )

        self.tx.publish(host_msg)

        return

    def stop(self):
        pass
