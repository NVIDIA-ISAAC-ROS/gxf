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
from gxf.std import Receiver
from gxf.std import Tensor
from gxf.python_codelet import CodeletAdapter
import numpy as np


class VerifyTensor(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of TensorDescription.
    Receives a message on the Receiver on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(), self.cid(), self.params["receiver0"])
        self.count = 0

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

        numpy_array = np.asarray(tensor0)
        print("received complex tensor type: ", numpy_array.dtype)
        print("First 10 elements of received complex tensor: ", numpy_array[:10])

        return

    def stop(self):
        pass
