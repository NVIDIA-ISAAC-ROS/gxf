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
from gxf.core import exception_handler
import gxf.std as std
from gxf.sample import PingTx
from gxf.sample import PingRx
from gxf.std import StandardGraph
import io
import signal

import logging

class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stream = io.StringIO()

    def emit(self, record):
        msg = self.format(record)
        self.stream.write(msg + '\n')

    def get_value(self):
        return self.stream.getvalue()

class TestCore(unittest.TestCase):
    '''
    Test loading subgraph via the application API
    '''
    @classmethod
    def setUpClass(cls):
        # Set up the logger once for all tests
        cls.log_handler = StringIOHandler()
        cls.log_handler.setLevel(logging.INFO)
        logger = logging.getLogger("Core")
        logger.addHandler(cls.log_handler)
        logger.setLevel(logging.INFO)

    @classmethod
    def tearDownClass(cls):
        # Remove the handler after all tests
        logger = logging.getLogger("Core")
        logger.removeHandler(cls.log_handler)

    def setUp(self):
        # Clear the log buffer before each test
        self.log_handler.stream = io.StringIO()

    def tearDown(self):
        pass


    '''
    Test crash handler setup in Graph
    '''
    def test_crash_handler(self):
        class CrashGraph(StandardGraph):
            def __init__(self, name):
                super().__init__(name)
                tx = self.add_codelet(PingTx(clock=self.get_clock()), count=5)
                rx = self.add_codelet(PingRx())
                self.connect(tx.signal, rx.signal)

            # exception_handler should NOT take effect for signaled crash
            @exception_handler
            def run(self):
                signal.raise_signal(signal.SIGABRT)

        g = CrashGraph("crash_graph")
        g.load_extensions()

        with self.assertRaises(SystemExit):
            g.run()

        log_output = self.log_handler.get_value()
        self.assertIn("==== GXF Python Crash Handler ====", log_output)
        self.assertIn("Caught signal 6 (SIGABRT)", log_output)
        self.assertIn("==== Crash Handler Backtrace ====", log_output)

    '''
    Test exception handler decorator only
    applying the decorator to a new method, where raise an exception
    '''
    def test_exception_handler_decorator(self):
        class ExceptionGraphDec(StandardGraph):
            def __init__(self, name):
                super().__init__(name)
                tx = self.add_codelet(PingTx(clock=self.get_clock()), count=5)
                rx = self.add_codelet(PingRx())
                self.connect(tx.signal, rx.signal)

            @exception_handler
            def run(self):
                raise ValueError("Mock exception")

        g = ExceptionGraphDec("exception_decorator_graph")
        g.load_extensions()

        with self.assertRaises(SystemExit):
            g.run()  # new run()

        log_output = self.log_handler.get_value()
        self.assertIn("==== GXF Python Exception Handler ====", log_output)
        self.assertIn("Caught exception in run: ValueError: Mock exception", log_output)
        self.assertIn("==== Exception Handler Backtrace ====", log_output)

    '''
    Test exception handler for real exception from gxf
    Compose a wrong graph and try to find non exist component
    '''
    def test_exception_handler_real(self):
        class ExceptionGraphReal(StandardGraph):
            def __init__(self, name):
                super().__init__(name)
                tx = self.add_codelet(PingTx(clock=self.get_clock()), count=5)
                rx = self.add_codelet(PingTx())  # wrong graph
                self.connect(tx.signal, rx.signal)  # rx has no signal component

        g = ExceptionGraphReal("real_exception_graph")
        g.load_extensions()

        with self.assertRaises(SystemExit):
            g.run()  # base class run()

        log_output = self.log_handler.get_value()
        self.assertIn("==== GXF Python Exception Handler ====", log_output)
        self.assertIn("Caught exception in run: ValueError: GXF_ENTITY_COMPONENT_NOT_FOUND", log_output)
        self.assertIn("==== Exception Handler Backtrace ====", log_output)


if __name__ == '__main__':
    unittest.main()
