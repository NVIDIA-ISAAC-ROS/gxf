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

from gxf.core import Graph
from gxf.core import MessageEntity
from gxf.core import Entity
import gxf.std as std
from gxf.std import DoubleBufferTransmitter, DoubleBufferReceiver
from gxf.std import DownstreamReceptiveSchedulingTerm
from gxf.std import MessageAvailableSchedulingTerm
from gxf.std import CountSchedulingTerm
from gxf.std import Clock, RealtimeClock
from gxf.std import GreedyScheduler
from gxf.std import Transmitter, Receiver
from gxf.python_codelet import PyData
from gxf.python_codelet import PyCodeletV0, CodeletAdapter


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
        python_data = np.array([1, 2, 3, 4])
        PyData.add_to_entity(msg, python_data)
        self.tx.publish(msg, 1)
        return

    def stop(self):
        pass


class PyPingRx(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of Ping Rx.
    Receives a message on the Receiver on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(),\
                                self.cid(),\
                                self.params['receiver'])
        self.clock = Clock.get(self.context(), self.cid(), self.params['clock'])
        return

    def tick(self):
        msg = self.rx.receive()
        data = PyData.get_from_entity(msg)
        print(data)
        assert(np.array_equal(data, np.array([1, 2, 3, 4])))
        if(not msg):
            raise Exception("Didn't get message")
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

    def test_python_data_transmittion(self):
        g =  Graph()

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
        prx.add(PyCodeletV0(name='pingrx', codelet=PyPingRx, receiver= prx.dbr, clock=clock))
        # prx.add(PingRx(name='pingrx', signal=prx.dbr))
        std.connect(ptx.dbt, prx.dbr)
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))

        g.load_extensions()
        g.run()
        g.destroy()



if __name__ == '__main__':
    unittest.main()
