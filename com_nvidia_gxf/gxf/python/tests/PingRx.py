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
from gxf.std import Receiver
from gxf.std import Clock


class PingRx(CodeletAdapter):
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
        if(not msg):
            raise Exception("Didn't get message")
        return

    def stop(self):
        pass
