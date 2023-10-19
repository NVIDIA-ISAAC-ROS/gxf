"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from gxf.python_codelet import CodeletAdapter
from gxf.core import MessageEntity
from gxf.std import Clock
from gxf.std import Transmitter
from gxf.std import TargetTimeSchedulingTerm


class ScheduledPingTx(CodeletAdapter):
    """ Python codelet to send a msg on tick() at a specific time

    Sends a ping after a user-defined delay.
    """

    def start(self):
        self.params = self.get_params()
        self.tx = Transmitter.get(
            self.context(), self.cid(), self.params['transmitter'])
        self.execution_clock = Clock.get(
            self.context(), self.cid(), self.params['execution_clock'])
        self.target_time_scheduling_term = TargetTimeSchedulingTerm.get(
            self.context(), self.cid(), self.params['scheduling_term'])
        self.delay = int(self.params['delay'])
        self.target_time_scheduling_term.set_next_target_time(self.get_execution_timestamp())
        return

    def tick(self):
        target_timestamp = self.execution_clock.timestamp() + self.delay
        self.target_time_scheduling_term.set_next_target_time(target_timestamp)
        msg = MessageEntity(self.context())
        self.tx.publish(msg, 1)
        return

    def stop(self):
        pass
