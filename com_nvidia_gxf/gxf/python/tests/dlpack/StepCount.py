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
from gxf.python_codelet import CodeletAdapter


class StepCount(CodeletAdapter):
    """Python codelet to count the number of ticks()

    Python implementation of StepCount.
    User can set the expected number of ticks()
    and after running is the count matches this number
    it passes else raises an exception.
    """

    def start(self):
        self.params = self.get_params()
        self.expected_count = int(self.params["expected_count"])
        self.current_count = 0

    def tick(self):
        self.current_count += 1
        return

    def stop(self):
        if self.current_count != self.expected_count:
            raise Exception(
                "Codelet didn't tick expected number of times!"
                "\nExpected tick count = {}\nCodelet tick count = {}\n".format(
                    self.expected_count, self.current_count
                )
            )
        else:
            pass
