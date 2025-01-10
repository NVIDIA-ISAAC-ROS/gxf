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
import cupy as cp
import numpy as np

from gxf.python_codelet import CodeletAdapter
from gxf.std import Receiver, Tensor


class VerifyEqual(CodeletAdapter):
    """Python codelet to compare a GPU tensor to a CPU tensor to within a tolerance.

    Receives messages on receiver0 and receiver1 Receivers on every tick()

    If `use_array_interface` the `__array_interface__` or `__cuda_array_interface` are used,
    otherwise DLPack is used.
    """

    def start(self):
        self.params = self.get_params()
        self.rxs = [
            Receiver.get(self.context(), self.cid(), self.params[f"receiver{i}"])
            for i in range(0, 2)
        ]
        self.use_array_interface = bool(self.params.get("use_array_interface", False))

    def tick(self):
        in_host = self.rxs[0].receive()
        in_dev = self.rxs[1].receive()

        # Get all the tensors from the message.
        tensor_dev = Tensor.find_all_from_entity(in_dev)
        np.testing.assert_equal(len(tensor_dev), 1)
        tensor_host = Tensor.find_all_from_entity(in_host)
        np.testing.assert_equal(len(tensor_host), 1)

        if self.use_array_interface:
            dev_array = cp.asarray(tensor_dev[0])
            host_array = np.asarray(tensor_host[0])
        else:
            dev_array = cp.from_dlpack(tensor_dev[0])
            host_array = np.from_dlpack(tensor_host[0])
        # print(f"VerifyEqual: {dev_array[0, :4]=}")
        # print(f"VerifyEqual: {host_array[0, :4]=}")
        cp.testing.assert_allclose(dev_array, host_array, rtol=1e-5)
        return

    def stop(self):
        pass
