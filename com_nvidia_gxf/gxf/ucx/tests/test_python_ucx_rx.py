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
import os

from gxf.core import Graph
import gxf.std as std
from gxf.std import BlockMemoryPool
from gxf.std import DoubleBufferReceiver
from gxf.std import MessageAvailableSchedulingTerm
from gxf.std import CountSchedulingTerm
from gxf.std import ManualClock, RealtimeClock
from gxf.std import GreedyScheduler
from gxf.std import Entity
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.std import Transmitter, Receiver
from gxf.python_codelet import PyData
from gxf.python_codelet import PyComputeEntity
from gxf.std import Tensor

from gxf.ucx import UCX, UCXSource

import logging

class PyPingRx(CodeletAdapter):
    """ Python codelet to send a msg on tick()

    Python implementation of Ping Tx.
    Sends a message to the transmitter on every tick()
    """

    def __init__(self):
      super().__init__()
      self.rxs = ["rx"]

    def start(self):
        self.params = self.get_params()

        self.shape_expected = self.params.get("shape_expected", [1, 2])
        self.expected_ones = cp.ones(self.shape_expected)
        self.expected_zeros = cp.zeros(self.shape_expected)

    def tick(self):
        msg = self.rx.receive()

        ones_tensor = Tensor.get_from_entity(msg, "ones_tensor")
        actual_ones = cp.asarray(ones_tensor)
        cp.testing.assert_allclose(actual_ones, self.expected_ones, rtol=1e-5)
        print("Correctly received tensor from remote CuPy over UCX:\n" + str(actual_ones))

        zeros_tensor = Tensor.get_from_entity(msg, "zeros_tensor")
        actual_zeros = cp.asarray(zeros_tensor)
        cp.testing.assert_allclose(actual_zeros, self.expected_zeros, rtol=1e-5)
        print("Correctly received tensor from remote Allocator over UCX:\n" + str(actual_zeros))

        return

    def stop(self):
        return

class TestCore(unittest.TestCase):
    '''
    Test loading subgraph via the application API
    '''
    @classmethod
    def setUpClass(cls):
        """method will be ran once before any test is ran"""
        pass

    @classmethod
    def tearDownClass(cls):
        """method will be ran once after all tests have run"""
        pass

    def setUp(self):
        """ran before each test"""
        return super().setUp()

    def tearDown(self):
        """ran after each test"""
        return super().tearDown()

    def test_python_graph_python_codelet(self):
        g = Graph()
        g.set_severity(logging.DEBUG)
        clock = std.set_clock(g, RealtimeClock(name="clock"))
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
        g.add(UCX("ucx", allocator=g.mem_pool.device_image_pool))

        g.add(UCXSource("source", address="localhost", port=13338))
        g.add(PyComputeEntity("PingRx", count=-1)).add_codelet(
            "pingrx",
            PyPingRx(),
            allocator=g.mem_pool.device_image_pool,
            shape_expected=[2, 3],
        )

        std.connect(g.source.output, g.PingRx.rx)

        g.load_extensions()
        g.run()
        g.destroy()

if __name__ == '__main__':
    unittest.main()