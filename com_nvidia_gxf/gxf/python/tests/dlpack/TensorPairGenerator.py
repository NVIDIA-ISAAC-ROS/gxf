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
import os

import cupy as cp
import numpy as np

from gxf.core import MessageEntity
from gxf.python_codelet import CodeletAdapter
from gxf.std import Tensor, Transmitter

os.environ["CUPY_CACHE_IN_MEMORY"] = "1"


class TensorPairGenerator(CodeletAdapter):
    """Python codelet to generate a stream of tensors on tick()

    Generates a pair of NumPy tensors on transmitter0
    Generates a pair of CuPy tensors on transmitter1

    If `use_array_interface` the `__array_interface__` and `__cuda_array_interface` are used,
    otherwise DLPack is used.
    """

    def start(self):
        self.params = self.get_params()
        self.txs = [
            Transmitter.get(self.context(), self.cid(), self.params[f"transmitter{i}"])
            for i in range(0, 2)
        ]
        self.use_array_interface = bool(self.params.get("use_array_interface", False))
        rows = self.params.get("rows", 16)
        cols = self.params.get("cols", 64)
        self.shape = (rows, cols)
        self.dtype = np.dtype(self.params.get("dtype", np.float32))
        return

    def tick(self):
        dev_msg = MessageEntity(self.context())
        host_msg = MessageEntity(self.context())

        np1 = np.arange(self.shape[0] * self.shape[1], dtype=self.dtype).reshape(self.shape)
        np2 = np.ascontiguousarray(np1.transpose())
        # print(f"Generator: {np1[0, :4]=}")
        # print(f"Generator: {np2[0, :4]=}")
        for i, arr in enumerate([np1, np2]):
            if self.use_array_interface:
                Tensor.add_to_entity(
                    host_msg, Tensor.as_tensor(arr), f"host_tensor{i + 1}"
                )
            else:
                Tensor.add_to_entity(host_msg, Tensor.from_dlpack(arr), f"host_tensor{i + 1}")

        cp1 = cp.asarray(np1)
        cp2 = cp.asarray(np2)
        # print(f"Generator: {cp1[0, :4]=}")
        # print(f"Generator: {cp2[0, :4]=}")
        for i, dev_arr in enumerate([cp1, cp2]):
            if self.use_array_interface:
                Tensor.add_to_entity(
                    dev_msg, Tensor.as_tensor(dev_arr), f"cuda_tensor{i + 1}"
                )
            else:
                Tensor.add_to_entity(dev_msg, Tensor.from_dlpack(dev_arr), f"cuda_tensor{i + 1}")

        self.txs[0].publish(host_msg)
        self.txs[1].publish(dev_msg)

        return

    def stop(self):
        pass
