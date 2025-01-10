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
import numpy as np

from gxf.std import Clock
from gxf.std import Receiver
from gxf.std import Tensor
from gxf.python_codelet import CodeletAdapter

class VerifyTensorDescription(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of TensorDescription.
    Receives a message on the Receiver on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(),\
                                self.cid(),\
                                self.params['receiver'])
        self.clock = Clock.get(self.context(), self.cid(), self.params['clock'])
        self.count = 0

    def tick(self):
        in0 = self.rx.receive()

        # method 1 to get all the tensors from the message.
        # In this case we only have one tensor hence tensor0
        # is the first element.
        tensor0 = Tensor.find_all_from_entity(in0)
        tensor0 = tensor0[0]

        SHAPE_SIZE=1024
        SHAPE_RANK=2
        BYTES_PER_ELEMENT=4

        # The values are from CreateTensor.py, should be same as host_tensor
        np.testing.assert_equal(tensor0.get_tensor_description().shape.size(), SHAPE_SIZE)
        np.testing.assert_equal(tensor0.get_tensor_description().shape.rank(), SHAPE_RANK)
        np.testing.assert_equal(tensor0.get_tensor_description().bytes_per_element,
                                BYTES_PER_ELEMENT)

        return

    def stop(self):
        pass