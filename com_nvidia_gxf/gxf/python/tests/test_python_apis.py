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
import unittest
import numpy as np
import os

from gxf.core import EntityGroup
from gxf.core import Graph
from gxf.core import Component
import gxf.std as std
from gxf.std import DoubleBufferTransmitter
from gxf.std import DoubleBufferReceiver
from gxf.std import DownstreamReceptiveSchedulingTerm
from gxf.std import MessageAvailableSchedulingTerm
from gxf.std import CountSchedulingTerm
from gxf.std import RealtimeClock
from gxf.std import ManualClock
from gxf.std import GreedyScheduler
from gxf.std import Entity
from gxf.std import ComputeEntity
from gxf.std import CPUThread
from gxf.std import GPUDevice
from gxf.std import Forward
from gxf.std import Broadcast
from gxf.std import Gather
from gxf.std import MultiThreadScheduler
from gxf.std import ThreadPool
from gxf.sample import PingBatchRx
from gxf.sample import PingTx
from gxf.sample import PingRx
from gxf.std import Subgraph
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.std import Transmitter
from gxf.python_codelet import PyCodeletV0


from gxf.core import parameter_get_str
import logging

APP_YAML = "gxf/python/tests/test_python_apis.yaml"
MANIFEST_YAML = "gxf/python/tests/test_python_apis_manifest.yaml"


class PyPingTx(CodeletAdapter):
    """ Python codelet to send a msg on tick()

    Python implementation of Ping Tx.
    Sends a message to the transmitter on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.tx = Transmitter.get(self.context(),\
                                    self.cid(),\
                                    self.params["transmitter"])
        pass

    def tick(self):
        msg = MessageEntity(self.context())
        self.tx.publish(msg, 1)
        return

    def stop(self):
        pass


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

    def test_python_graph_apis(self):
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        ptx = g.add(Entity("PingTx"))
        ptx.add(DoubleBufferTransmitter(name='dbt'))
        ptx.add(DownstreamReceptiveSchedulingTerm(name='drst',
                                                  transmitter=ptx.dbt,
                                                  min_size=1))
        ptx.add(PingTx(name='pingtx', signal=ptx.dbt, clock=clock))
        ptx.add(CountSchedulingTerm(name='cst', count=5))

        prx = g.add(Entity("PingRx"))
        prx.add(DoubleBufferReceiver(name='dbr'))
        prx.add(MessageAvailableSchedulingTerm(
            name='mast', receiver=prx.dbr, min_size=1))
        prx.add(PingRx(name='pingrx', signal=prx.dbr))
        std.connect(g.PingTx.dbt, prx.dbr)
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        g.load_extensions()
        g.run()
        g.destroy()

    def test_python_entity_impl(self):
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

    def test_python_subgraph_component(self):
        # system
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        g.add(ComputeEntity("tx1", count=10)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("tx2", count=10)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("rx")).add_codelet(PingBatchRx(
            name="codelet", batch_size=2), min_message_available=1, rx_capacity=2)

        sg = g.add(Entity("forward_subgraph"))
        sg.add(Subgraph(name="forward_subgraph",
                        location="gxf/python/tests/gather_subgraph.yaml",
                        prerequisites={"monitored_rx": g.rx.signal},
                        override_params={
                            "output": {"capacity": 2},
                            "forward1_buf_term": {"min_size": 1}
                        }))

        c = g.add(Entity("connections"))
        c.add(Component(name='c1', type="nvidia::gxf::Connection",
              source=g.tx1.signal, target="forward_subgraph/input1"))
        c.add(Component(name='c2', type="nvidia::gxf::Connection",
              source=g.tx2.signal, target="forward_subgraph/input2"))
        c.add(Component(name='c3', type="nvidia::gxf::Connection",
              source="forward_subgraph/output", target=g.rx.signal))

        g.load_extensions()
        g.save("/tmp/graph.yaml")
        g.run()
        g.destroy()

    def test_python_subgraph2(self):
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))
        std.enable_job_statistics(g, clock=clock)

        transmitter = g.add(Entity("transmitter"))
        transmitter.add(DoubleBufferTransmitter(name='tx_signal'))
        transmitter.add(
            PingTx(name='pingtx', signal=transmitter.tx_signal, clock=clock))
        transmitter.add(
            DownstreamReceptiveSchedulingTerm(name='st', transmitter=transmitter.tx_signal, min_size=1))
        transmitter.add(CountSchedulingTerm(name='cst', count=5))

        receiver = g.add(Entity("receiver"))
        receiver.add(
            Subgraph(name='receiver', location="gxf/python/tests/receiver_subgraph.yaml"))

        connection = g.add(Entity("connections2", True))
        connection.add(
            Component(name="connection",
                      type="nvidia::gxf::Connection",
                      source="transmitter/tx_signal",
                      target="receiver/rx_signal"))

        g.load_extensions()
        g.save("/tmp/graph.yaml")
        g.run()
        g.destroy()

    def test_python_entity_group(self):
        # system
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, MultiThreadScheduler(
            max_duration_ms=1000, clock=clock, stop_on_deadlock=False, worker_thread_number=2))

        g.add(ComputeEntity("tx0", count=10)).add_codelet(
            PingTx(name="ping_tx_c0", clock=clock))
        g.tx0.add(CPUThread(name="cpu_thread", pin_entity=True))
        g.add(ComputeEntity("rx0")).add_codelet(PingRx(name="ping_rx_c0"))
        g.rx0.add(CPUThread(name="cpu_thread", pin_entity=True))
        g.add(ComputeEntity("tx1", count=10)).add_codelet(
            PingTx(name="ping_tx_c1"))
        g.tx1.add(CPUThread(name="cpu_thread", pin_entity=True))
        g.add(ComputeEntity("rx1")).add_codelet(PingRx(name="ping_rx_c0"))
        g.rx1.add(CPUThread(name="cpu_thread", pin_entity=True))

        std.connect(g.tx0.signal, g.rx0.signal)
        std.connect(g.tx1.signal, g.rx1.signal)

        g.add(Entity("GPU_0")).add(GPUDevice(name="GPU_0", dev_id=0))
        g.add(Entity("GPU_1")).add(GPUDevice(name="GPU_1", dev_id=1))
        g.add(Entity("GPU_2")).add(GPUDevice(name="GPU_2", dev_id=2))
        g.add(Entity("CPU_0")).add(ThreadPool(name="ThP_0", initial_size=1))
        g.add(Entity("CPU_1")).add(ThreadPool(name="ThP_1", initial_size=1))
        g.add(Entity("CPU_2")).add(ThreadPool(name="ThP_2", initial_size=1))

        eg0 = g.add(EntityGroup(name="EG_0"))
        eg0.add([g.tx0, g.tx1, g.GPU_1, g.CPU_1])

        eg0 = g.add(EntityGroup(name="EG_1"))
        eg0.add([g.rx0, g.rx1, g.GPU_2, g.CPU_2])

        g.load_extensions()
        g.run()
        g.destroy()


    def receiver_subgraph(self):
        sg = Graph("receiver")
        # sg.add(ComputeEntity("rx")).add_codelet(PingRx())
        e = sg.add(Entity(name="rx"))
        e.add(DoubleBufferReceiver(name="signal"))
        e.add(PingRx(name="pingrx", signal=e.signal), visible_as="codelet")
        e.add(MessageAvailableSchedulingTerm(name="mast", receiver=e.signal, min_size=1))
        return sg

    def test_python_subgraph(self):
        g = Graph(name='')
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))
        std.enable_job_statistics(g, clock=clock)

        transmitter = g.add(Entity("transmitter"))
        transmitter.add(DoubleBufferTransmitter(name='tx_signal'))
        transmitter.add(
            PingTx(name='pingtx', signal=transmitter.tx_signal, clock=clock))
        transmitter.add(
            DownstreamReceptiveSchedulingTerm(name='st', transmitter=transmitter.tx_signal, min_size=1))
        transmitter.add(CountSchedulingTerm(name='cst', count=5))

        sg = g.add(self.receiver_subgraph())

        std.connect(transmitter.tx_signal, sg.rx.signal, g)

        g.load_extensions()
        g.run()
        g.destroy()

    def transmitter_subgraph(self, clock):
        sg = Graph("transmitter")
        transmitter = sg.add(Entity("transmitter"))
        transmitter.add(DoubleBufferTransmitter(name='tx_signal'), visible_as="tx_signal")
        transmitter.add(
            PingTx(name='pingtx', signal=transmitter.tx_signal, clock = clock), visible_as="codelet")
        transmitter.add(
            DownstreamReceptiveSchedulingTerm(name='st', transmitter=transmitter.tx_signal, min_size=1))
        transmitter.add(CountSchedulingTerm(name='cst', count=5))
        return sg

    def test_python_subgraph_params(self):
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))
        std.enable_job_statistics(g, clock=clock)

        e = g.add(Entity(name="rx"))
        e.add(DoubleBufferReceiver(name="signal"))
        e.add(PingRx(name="pingrx", signal=e.signal))
        e.add(MessageAvailableSchedulingTerm(name="mast", receiver=e.signal, min_size=1))
        sg = g.add(self.transmitter_subgraph(clock))
        std.connect(sg.get("tx_signal"), e.signal, g)

        g.load_extensions()
        g.save("/tmp/graph.yaml")
        g.run()
        g.destroy()


    def receiver_subgraph2(self, name="receiver"):
        sg = Graph(name)
        e = sg.add(Entity(name="rx"))
        e.add(DoubleBufferReceiver(name="signal"), visible_as="rx_signal")
        e.add(PingRx(name="pingrx", signal=e.signal))
        e.add(MessageAvailableSchedulingTerm(name="mast", receiver=e.signal, min_size=1))
        return sg

    def test_python_subgraph_params2(self):
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))
        std.enable_job_statistics(g, clock=clock)

        transmitter = g.add(Entity("transmitter"))
        transmitter.add(DoubleBufferTransmitter(name='tx_signal'))
        transmitter.add(
            PingTx(name='pingtx', signal=transmitter.tx_signal, clock = clock), visible_as="codelet")
        transmitter.add(
            DownstreamReceptiveSchedulingTerm(name='st', transmitter=transmitter.tx_signal, min_size=1))
        transmitter.add(CountSchedulingTerm(name='cst', count=5))


        broadcast = g.add(Entity(name="broadcast"))
        broadcast.add(DoubleBufferReceiver(name='brx'))
        broadcast.add(DoubleBufferTransmitter(name='btx1'))
        broadcast.add(DoubleBufferTransmitter(name='btx2'))
        broadcast.add(MessageAvailableSchedulingTerm(name='mast', receiver=broadcast.brx, min_size=1))
        broadcast.add(Broadcast(name="bcst", source=broadcast.brx))

        sg1 = g.add(self.receiver_subgraph2(name="rx1"))
        sg2 = g.add(self.receiver_subgraph2(name="rx2"))
        std.connect(transmitter.tx_signal, broadcast.brx, g)
        std.connect(broadcast.btx1, sg1.get("rx_signal"), g)
        std.connect(broadcast.btx2, sg2.get("rx_signal"), g)

        g.load_extensions()
        g.save("/tmp/graph.yaml")
        g.run()
        g.destroy()

    def get_forward_subgraph(self, name, monitored_rx):
        g = Graph(name=name)
        g.add(ComputeEntity("block1")).add_codelet(Forward())
        g.add(ComputeEntity("block2")).add_codelet(Forward(), min_message_available=1000)
        g.add(ComputeEntity("block3")).add_codelet(Forward())
        std.connect(g.block1.out, getattr(g.block2, "in"))
        std.connect(g.block2.out, getattr(g.block3, "in"))
        g.make_visible("receiver", getattr(g.block1, "in"))
        g.make_visible("transmitter", g.block3.out)
        g.make_visible("buffer_term", g.block2.mast)
        return g


    def get_gather_subgraph(self, name, monitored_rx):
        g = Graph(name=name)
        g.add(ComputeEntity("i1")).add_codelet(Forward())
        g.add(ComputeEntity("i2")).add_codelet(Forward())
        g.add(self.get_forward_subgraph("forward_subgraph_1", monitored_rx))
        g.add(self.get_forward_subgraph("forward_subgraph_2", monitored_rx))
        g.add(ComputeEntity("gather")).add_codelet(Gather())
        g.gather.add(DoubleBufferReceiver(name="input1"))
        g.gather.add(DoubleBufferReceiver(name="input2"))
        g.gather.add(MessageAvailableSchedulingTerm(name="mast1", receiver=g.gather.input1, min_size=1))
        g.gather.add(MessageAvailableSchedulingTerm(name="mast2", receiver=g.gather.input2, min_size=1))
        std.connect(g.i1.out, g.forward_subgraph_1.get("receiver"), g)
        std.connect(g.i2.out, g.forward_subgraph_2.get("receiver"), g)
        std.connect(g.forward_subgraph_1.get("transmitter"), g.gather.input1, g)
        std.connect(g.forward_subgraph_2.get("transmitter"), g.gather.input2, g)
        g.make_visible("input1", getattr(g.i1, "in"))
        g.make_visible("input2", getattr(g.i2, "in"))
        g.make_visible("output", g.gather.sink)
        g.make_visible("forward1_buf_term", g.forward_subgraph_1.get("buffer_term"))
        g.forward_subgraph_2.get("buffer_term").set_param("min_size", 1)
        return g

    def test_python_subgraph_multi_level(self):
        g =  Graph()
        g.set_severity(logging.WARN)
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        g.add(ComputeEntity("tx1", count=10)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("tx2", count=10)).add_codelet(PingTx(clock=clock))
        g.add(ComputeEntity("rx")).add_codelet(PingBatchRx(
            name="codelet", batch_size=2), min_message_available=1, rx_capacity=2)

        gather = g.add(self.get_gather_subgraph("gather_subgraph", monitored_rx=g.rx.signal))

        gather.get("output").set_param("capacity", 2)
        gather.get("forward1_buf_term").set_param("min_size", 1)

        std.connect(g.tx1.signal, gather.get("input1"), g)
        std.connect(g.tx2.signal, gather.get("input2"), g)
        std.connect(gather.get("output"), g.rx.signal, g)
        g.load_extensions(extension_filenames=['gxf/std/libgxf_std.so'])
        g.load_extensions(extension_filenames=['sample/libgxf_sample.so'], workspace='gxf')
        g.run()
        g.destroy()

    def test_python_graph_python_codelet(self):

        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name='clock'))
        ptx = g.add(Entity("PingTx"))
        ptx.add(DoubleBufferTransmitter(name='dbt'))
        ptx.add(DownstreamReceptiveSchedulingTerm(name='drst',
                                                  transmitter=ptx.dbt,
                                                  min_size=1))
        ptx.add(PyCodeletV0(name='pingtx', codelet=PyPingTx, transmitter= ptx.dbt))
        ptx.add(CountSchedulingTerm(name='cst', count=5))

        prx = g.add(Entity("PingRx"))
        prx.add(DoubleBufferReceiver(name='dbr'))
        prx.add(MessageAvailableSchedulingTerm(
            name='mast', receiver=prx.dbr, min_size=1))
        prx.add(PingRx(name='pingrx', signal=prx.dbr))
        std.connect(g.PingTx.dbt, prx.dbr)
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        g.load_extensions()
        g.run()
        g.destroy()


if __name__ == '__main__':
    unittest.main()
