"""
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.std import PeriodicSchedulingTerm
from gxf.std import Receiver
from gxf.std import Clock

class TestPeriodicSchedulingTerm(CodeletAdapter):
    """ Python codelet to receive a msg on tick()

    Python implementation of CountSchedulingTerm.
    Receives a message on the Reciever on every tick()
    """

    def start(self):
        self.params = self.get_params()
        self.execution_count = self.get_execution_count()
        self.rx = Receiver.get(self.context(), self.cid(), self.params["rx"])
        self.clock = Clock.get(
            self.context(), self.cid(), self.params["clock"])
        self.periodic_scheduling_term = PeriodicSchedulingTerm.get(
            self.context(), self.cid(), self.params["periodic_scheduling_term"])
        self.recess_time = self.periodic_scheduling_term.recess_period_ns()
        self.last_tick_ns = 0
        return

    def tick(self):
        msg = self.rx.receive()
        if self.get_execution_count() == 1:
            self.last_tick_ns = 0
        else:
            assert(self.get_execution_timestamp() == self.last_tick_ns + self.recess_time)
            self.last_tick_ns = self.get_execution_timestamp()
        assert(self.last_tick_ns == self.periodic_scheduling_term.last_run_timestamp())
        return

    def stop(self):
        pass
