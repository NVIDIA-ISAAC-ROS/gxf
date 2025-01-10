"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gxf.std as std
from gxf.std import DoubleBufferTransmitter
from gxf.std import DoubleBufferReceiver
from gxf.std import DownstreamReceptiveSchedulingTerm
from gxf.std import MessageAvailableSchedulingTerm
from gxf.std import CountSchedulingTerm
from gxf.std import RealtimeClock
from gxf.std import Clock
from gxf.std import GreedyScheduler
from gxf.std import Entity
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.std import Transmitter, Receiver
from gxf.python_codelet import PyCodeletV0
from gxf.std import BlockMemoryPool
from gxf.std import MemoryStorageType
from gxf.multimedia import VideoBuffer
from gxf.std import Allocator, Timestamp
from PIL import Image

import ctypes
import cupy

def get_cupy_ndarray_from_video_buffer(video_buffer):
    if video_buffer.storage_type() != MemoryStorageType.kDevice:
        raise RuntimeError(
            "The video_buffer should be on device and not on host!")
    # print(video_buffer.get_info())
    data_ptr, data_size, data_type, shape, strides = video_buffer.get_info()
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object, ctypes.c_char_p]
    unowned_mem_ptr = cupy.cuda.UnownedMemory(
        ctypes.pythonapi.PyCapsule_GetPointer(data_ptr, None), data_size, None)
    mem_ptr = cupy.cuda.MemoryPointer(unowned_mem_ptr, 0)
    cupy_array = cupy.ndarray(
        shape=shape, dtype=data_type, memptr=mem_ptr, strides=strides, order='C')
    return cupy_array

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
        self.device_allocator = Allocator.get(
            self.context(), self.cid(), self.params["device_allocator"]
        )
        pass

    def tick(self):
        msg = MessageEntity(self.context())
        image = Image.new(mode="RGBA", size=(1920, 1080), color=(118,185,0))
        vb = VideoBuffer(image, self.device_allocator, MemoryStorageType.kDevice)
        # print(vb.get_info())
        VideoBuffer.add_to_entity(msg, vb)
        Timestamp.add_to_entity(msg, 1000000, 6000000, "Timestamp")

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
                                self.params["receiver"])
        self.clock = Clock.get(self.context(), self.cid(), self.params["clock"])
        return

    def tick(self):
        msg = self.rx.receive()
        image = Image.new(mode="RGBA", size=(1920, 1080), color=(118,185,0))
        video_buffer = VideoBuffer.get_from_entity(msg)
        cupy_array = get_cupy_ndarray_from_video_buffer(video_buffer)
        # print(np.array(image))
        # print(np.array(video_buffer))
        assert np.array_equal(image, Image.fromarray(cupy_array.get())), "Error"
        if(not msg):
            raise Exception("Didn't get message")

        # Get timestamp from received message
        acqtime, pubtime = Timestamp.get_from_entity(msg, "Timestamp")
        assert acqtime == 1000000, "Recieved acqtime is incorrect"
        assert pubtime == 6000000, "Recieved pubtime is incorrect"

        return

    def stop(self):
        pass

class TestCore(unittest.TestCase):
    """
    Test loading subgraph via the application API
    """
    @classmethod
    def setUpClass(cls):
        # method will be ran once before any test is ran
        pass

    @classmethod
    def tearDownClass(cls):
        # method will be ran once after all tests have run
        pass

    # def setUp(self):
    #     # ran before each test
    #     return super().setUp()

    # def tearDown(self):
    #     # ran after each test
    #     return super().tearDown()

    def test_python_graph_python_codelet(self):
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name="clock"))
        g.add(Entity("mem_pool"))
        g.mem_pool.add(
            BlockMemoryPool(
                "device_image_pool",
                storage_type=1,
                block_size=1920* 1080 * 4*10,
                num_blocks=20,
            )
        )
        ptx = g.add(Entity("PingTx"))
        ptx.add(DoubleBufferTransmitter(name="dbt"))
        ptx.add(DownstreamReceptiveSchedulingTerm(name="drst",
                                                transmitter=ptx.dbt,
                                                min_size=1))
        ptx.add(PyCodeletV0(name="pingtx", codelet=PyPingTx,
                            transmitter= ptx.dbt,
                            device_allocator=g.mem_pool.device_image_pool))
        ptx.add(CountSchedulingTerm(name="cst", count=5))

        prx = g.add(Entity("PingRx"))
        prx.add(DoubleBufferReceiver(name="dbr"))
        prx.add(MessageAvailableSchedulingTerm(
            name="mast", receiver=prx.dbr, min_size=1))
        prx.add(PyCodeletV0(name="pingrx", codelet=PyPingRx,
                            receiver= prx.dbr,
                            clock=clock))
        std.connect(g.PingTx.dbt, prx.dbr)
        std.enable_job_statistics(g, clock=clock)
        std.set_scheduler(g, GreedyScheduler(
            max_duration_ms=1000000, clock=clock))
        g.load_extensions()
        g.run()
        g.destroy()

if __name__ == '__main__':
    unittest.main()
