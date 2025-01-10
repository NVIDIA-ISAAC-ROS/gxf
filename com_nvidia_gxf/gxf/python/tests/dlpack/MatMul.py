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

from gxf.core import MessageEntity
from gxf.python_codelet import CodeletAdapter
from gxf.std import Receiver, Tensor, Transmitter


class MatMul(CodeletAdapter):
    """Python codelet to multiply a pair of tensors and multiplies them.

    Receives a pair of Tensors on every tick.
    Transmits a Tensor on every tick.

    If `use_array_interface` the `__array_interface__` or `__cuda_array_interface` are used,
    otherwise DLPack is used.

    The `device` parameter can be set to either 'cpu' or 'gpu'.
    (alternatively we could automatically set this based on Tensor properties)
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(), self.cid(), self.params["rx"])
        self.tx = Transmitter.get(self.context(), self.cid(), self.params["tx"])
        self.use_array_interface = bool(self.params.get("use_array_interface", False))

        # use NumPy or CuPy based on the 'device' parameter
        device = self.params.get("device", "cpu")
        if (not isinstance(device, str) or device.lower() not in ["cpu", "gpu"]):
            raise ValueError("device parameter must be one of {'cpu', 'gpu'}")
        self.xp = cp if device == "gpu" else np

    def tick(self):
        xp = self.xp
        in_msg = self.rx.receive()
        in_tensors = Tensor.find_all_from_entity(in_msg)
        np.testing.assert_equal(len(in_tensors), 2)

        if self.use_array_interface:
            in0 = xp.asarray(in_tensors[0])
            in1 = xp.asarray(in_tensors[1])
        else:
            in0 = xp.from_dlpack(in_tensors[0])
            in1 = xp.from_dlpack(in_tensors[1])

        out = xp.dot(in0, in1)

        out_msg = MessageEntity(self.context())
        if self.use_array_interface:
            Tensor.add_to_entity(out_msg, Tensor.as_tensor(out), "")
        else:
            Tensor.add_to_entity(out_msg, Tensor.from_dlpack(out), "")
        self.tx.publish(out_msg)

    def stop(self):
        pass
