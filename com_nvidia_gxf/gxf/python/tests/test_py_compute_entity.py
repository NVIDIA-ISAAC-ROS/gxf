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
import numpy as np
from gxf.core import Graph
import gxf.std as std
from gxf.std import RealtimeClock
from gxf.std import GreedyScheduler
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.python_codelet import PyComputeEntity
from gxf.std import Tensor

class PyPingTx(CodeletAdapter):
    def __init__(self, some_param="what"):
        super().__init__()
        self.txs = ["tx"]
        self.some_param = some_param

    def start(self):
        self.params = self.get_params()
        self.count = 0
        pass

    def tick(self):
        msg = MessageEntity(self.context())
        t = Tensor(np.array([1+self.count, 2+self.count, 3+self.count]))
        Tensor.add_to_entity(msg, t)
        self.tx.publish(msg, 1)
        self.count+=1
        return

    def stop(self):
        pass


class PyPingRx(CodeletAdapter):
    def __init__(self):
        super().__init__()
        self.rxs = ["input"]

    def start(self):
        self.count = 0
        pass

    def tick(self):
        msg = self.input.receive()
        t = Tensor.get_from_entity(msg)
        assert(np.array_equal(np.array(t), [1+self.count, 2+self.count, 3+self.count]))
        self.count +=1
        return

    def stop(self):
        return


g = Graph()
clock = std.set_clock(g, RealtimeClock(name='clock'))

ptx = g.add(PyComputeEntity("PingTx", count=5))
ptx.add_codelet("somename", PyPingTx(some_param="some_value"))

prx = g.add(PyComputeEntity("PingRx", count = 5))
prx.add_codelet("codelet", PyPingRx())

std.connect(g.PingTx.tx, prx.input)

std.set_scheduler(g, GreedyScheduler(
    max_duration_ms=1000000, clock=clock))
g.load_extensions()
g.run()
g.destroy()
