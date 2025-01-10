"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
import numpy as np
import cupy as cp
import time
from gxf.core import Graph
import gxf.std as std
from gxf.std import Allocator
from gxf.std import BlockMemoryPool
from gxf.std import DoubleBufferTransmitter, Transmitter
from gxf.std import DownstreamReceptiveSchedulingTerm
from gxf.std import MemoryStorageType
from gxf.std import CountSchedulingTerm
from gxf.std import ManualClock, RealtimeClock
from gxf.std import GreedyScheduler
from gxf.std import Entity
from gxf.std import TensorDescription
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.python_codelet import PyComputeEntity
from gxf.std import Tensor
from gxf.std import Shape
from gxf.std import PrimitiveType
from gxf.ucx import UCX, UCXSink

import logging


class PyPingTx(CodeletAdapter):
    """ Python codelet to send a msg on tick()

    Python implementation of Ping Tx.
    Sends a message to the transmitter on every tick()
    """

    def __init__(self):
      super().__init__()
      self.txs = ["tx"]

    def start(self):
        self.params = self.get_params()

        self.allocator = Allocator.get(self.context(), self.cid(), self.params["allocator"])
        self.shape = self.params.get("shape", [1, 2])

    def tick(self):
        msg = MessageEntity(self.context())

        # add ones tensor allocated on cupy
        cp_tensor = cp.ones(self.shape)
        gxf_tensor = Tensor.as_tensor(cp_tensor)
        Tensor.add_to_entity(msg, gxf_tensor, "ones_tensor")
        # add uninitialized tensor allocated by gxf allocator
        tensor_on_allocator = Tensor.add_to_entity(msg, "zeros_tensor")
        td = TensorDescription(
            name="zeros_tensor",
            storage_type=MemoryStorageType.kDevice,
            shape=Shape(self.shape),
            element_type=PrimitiveType.kFloat32,
            bytes_per_element=4
        )
        tensor_on_allocator.reshape(td, self.allocator)

        self.tx.publish(msg, 1)
        return

    def stop(self):
        pass


class TestUCXAPIs(unittest.TestCase):
    """
    Test UCX APIs
    """
    @classmethod
    def setUpClass(cls):
        """method will be ran once before any test is ran
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """method will be ran once after all tests have run
        """
        pass

    def setUp(self):
        """ran before each test
        """
        return super().setUp()

    def tearDown(self):
        """ ran after each test
        """
        return super().tearDown()

    def test_python_graph_python_codelet(self):
        g = Graph()
        g.set_severity(logging.DEBUG)
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=5000, clock=clock, stop_on_deadlock=False))
        g.add(Entity("mem_pool")).add(
            BlockMemoryPool(
                "device_image_pool",
                storage_type=1,
                block_size=1920 * 1080 * 4,
                num_blocks=150,
            )
        )
        # g.add(UCX("ucx", allocator=g.mem_pool.device_image_pool))
        g.add(UCX("ucx"))

        g.add(PyComputeEntity("PingTx", count=5)).add_codelet(
            "pingtx",
            PyPingTx(),
            allocator=g.mem_pool.device_image_pool,
            shape=[2, 3],
        )
        g.add(UCXSink("sink", count=-1, address="localhost", port=13338))

        std.connect(g.PingTx.tx, g.sink.input)

        g.load_extensions()
        g.run()
        g.destroy()


if __name__ == '__main__':
    unittest.main()
