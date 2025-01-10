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
import cupy as cp
import numpy as np

from gxf.core import Graph
import gxf.std as std
from gxf.std import DoubleBufferTransmitter
from gxf.std import DoubleBufferReceiver
from gxf.std import DownstreamReceptiveSchedulingTerm
from gxf.std import MessageAvailableSchedulingTerm
from gxf.std import CountSchedulingTerm
from gxf.std import RealtimeClock
from gxf.std import GreedyScheduler
from gxf.std import Entity
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.std import Transmitter, Receiver
from gxf.python_codelet import PyCodeletV0
from gxf.std import Tensor


class PyPingTx(CodeletAdapter):
    """ Python codelet to send a msg on tick()

    Python implementation of Ping Tx.
    Sends a message to the transmitter on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.tx = Transmitter.get(self.context(),
                                  self.cid(),
                                  self.params["transmitter"])
        self.count = 0
        pass

    def tick(self):
        msg = MessageEntity(self.context())

        # add both host tensor and then device tensor
        for xp in [np, cp]:
            arr = xp.array([1 + self.count, 2 + self.count, 3 + self.count])

            # alternately test multiple valid ways to create the Tensor
            creation_mode = self.count % 2
            if creation_mode == 0 and xp == np:
                # Tensor() constructor is for pybind11::array which is specific to NumPy
                # (For CuPy with creation_mode == 0, we use as_tensor() again instead)
                t = Tensor(arr)
            elif creation_mode <= 1:
                t = Tensor.as_tensor(arr)
            elif creation_mode == 2:
                t = Tensor.from_dlpack(arr)

            # verify that TensorDescriptor has expected values
            np.testing.assert_equal(t.get_tensor_description().shape.rank(), arr.ndim)
            np.testing.assert_equal(t.get_tensor_description().shape.size(), arr.size)
            np.testing.assert_equal(t.get_tensor_description().bytes_per_element, arr.itemsize)

            name = "host_tensor" if xp == np else "device_tensor"
            Tensor.add_to_entity(msg, t, name)
        self.tx.publish(msg, 1)
        self.count += 1
        return

    def stop(self):
        pass


class PyPingRx(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of Ping Rx.
    Receive a message from the transmitter on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.rx = Receiver.get(self.context(),
                               self.cid(),
                               self.params["receiver"])
        self.count = 0
        pass

    def tick(self):
        msg = self.rx.receive()
        t = Tensor.get_from_entity(msg, "host_tensor")
        expected = [1 + self.count, 2 + self.count, 3 + self.count]
        np.testing.assert_array_equal(np.array(t), expected)

        # asarray should use the __array_interface__
        assert hasattr(t, '__array_interface__')
        np.testing.assert_array_equal(np.asarray(t), expected)

        # use from_dlpack to use the DLPack protocol
        np.testing.assert_array_equal(np.from_dlpack(t), expected)

        all_tensors = Tensor.find_all_from_entity(msg)
        assert len(all_tensors) == 2
        t2 = all_tensors[0]
        np.testing.assert_array_equal(np.array(t2), expected)

        # asarray should use the __array_interface__
        assert hasattr(t2, '__array_interface__')
        np.testing.assert_array_equal(np.asarray(t2), expected)

        # check CUDA array interface
        t2_gpu = all_tensors[1]
        assert hasattr(t2_gpu, '__cuda_array_interface__')
        cp.testing.assert_array_equal(cp.asarray(t2_gpu), expected)

        # use from_dlpack to use the DLPack protocol for both tensors
        np.testing.assert_array_equal(np.from_dlpack(t2), expected)
        cp.testing.assert_array_equal(cp.from_dlpack(t2_gpu), expected)

        self.count +=1
        return

    def stop(self):
        return


class TestCore(unittest.TestCase):
    '''
    Test loading subgraph via the application API
    '''
    @classmethod
    def setUpClass(cls):
        # method will be run once before any test is ran
        pass

    @classmethod
    def tearDownClass(cls):
        # method will be run once after all tests have run
        pass

    def setUp(self):
        # ran before each test
        return super().setUp()

    def tearDown(self):
        # ran after each test
        return super().tearDown()

    def test_python_graph_python_codelet(self):

        class PythonCodeletTest(Graph):
            def compose(self, *args, **kwargs):
                clock = std.set_clock(self, RealtimeClock(name='clock'))
                ptx = self.add(Entity("PingTx"))
                ptx.add(DoubleBufferTransmitter(name='dbt'))
                ptx.add(DownstreamReceptiveSchedulingTerm(name='drst',
                                                          transmitter=ptx.dbt,
                                                          min_size=1))
                ptx.add(PyCodeletV0(name='pingtx', codelet=PyPingTx, transmitter= ptx.dbt))
                ptx.add(CountSchedulingTerm(name='cst', count=5))

                prx = self.add(Entity("PingRx"))
                prx.add(DoubleBufferReceiver(name='dbr'))
                prx.add(MessageAvailableSchedulingTerm(
                    name='mast', receiver=prx.dbr, min_size=1))
                prx.add(PyCodeletV0(name='pingrx', codelet=PyPingRx, receiver= prx.dbr))
                std.connect(self.PingTx.dbt, prx.dbr)
                std.enable_job_statistics(self, clock=clock)
                std.set_scheduler(self, GreedyScheduler(
                    max_duration_ms=1000000, clock=clock))
                return

        g = PythonCodeletTest()
        g.compose()
        g.load_extensions()
        g.run()
        g.destroy()

if __name__ == '__main__':
    unittest.main()
