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

from gxf.core import Graph
import gxf.std as std
from gxf.std import ManualClock
from gxf.std import GreedyScheduler
from gxf.std import Entity
from gxf.std import ComputeEntity
from gxf.sample import PingTx
from gxf.sample import PingRx
from gxf.sample import MultiPingRx

import logging

class TestCore(unittest.TestCase):
    '''
    Test loading subgraph via the application API
    '''
    @classmethod
    def setUpClass(cls):
        # method will be ran once before any test is ran
        pass

    @classmethod
    def tearDownClass(cls):
        # method will be ran once after all tests have run
        pass

    def setUp(self):
        # ran before each test
        return super().setUp()

    def tearDown(self):
        # ran after each test
        return super().tearDown()

    def test_compute_entity_for_scalar_tx_and_scalar_rx(self):
        # system
        g = Graph()
        clock = std.set_clock(g, ManualClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        # custom user code
        g.add(ComputeEntity("tx", count=5)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("rx")).add_codelet(PingRx())

        std.connect(g.tx.signal, g.rx.signal)

        g.load_extensions()
        g.run()
        g.destroy()

    def test_compute_entity_for_scalar_tx_and_list_rx(self):
        # system
        g = Graph()
        clock = std.set_clock(g, ManualClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        # custom user code
        g.add(ComputeEntity("tx_codelet_1", count=5)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("tx_codelet_2", count=5)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("tx_codelet_3", count=5)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("multi_rx_codelet")).add_codelet(MultiPingRx(), rx_num=3)

        std.connect(g.tx_codelet_1.signal, g.multi_rx_codelet.receivers[0])
        std.connect(g.tx_codelet_2.signal, g.multi_rx_codelet.receivers[1])
        std.connect(g.tx_codelet_3.signal, g.multi_rx_codelet.receivers[2])

        g.load_extensions()
        g.run()
        g.destroy()

    def test_compute_entity_for_scalar_tx_and_list_rx_single(self):
        # system
        g = Graph()
        clock = std.set_clock(g, ManualClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        # custom user code
        g.add(ComputeEntity("tx_codelet_1", count=5)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("multi_rx_codelet")).add_codelet(MultiPingRx())

        std.connect(g.tx_codelet_1.signal, g.multi_rx_codelet.receivers[0])

        g.load_extensions()
        g.run()
        g.destroy()

if __name__ == '__main__':
    unittest.main()