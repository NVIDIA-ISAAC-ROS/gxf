'''
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
'''

import gxf.benchmark as gxf_benchmark
from gxf.std import ComputeEntity
from gxf.core import Graph


class BenchmarkBuffer(Graph):
    """
    Benchmark buffer subgraph
    """

    def __init__(self, name):
        super().__init__(name)
        # Benchmark buffer
        buffer_entity = self.add(ComputeEntity('buffer_entity'))

        # Benchmark buffer components
        buffer_entity.add_codelet(gxf_benchmark.EntityBuffer(
            name = 'entity_buffer'), rx_capacity = 1)

        # Interface
        self.make_visible('rx', buffer_entity.receiver)
        self.make_visible('entity_buffer', buffer_entity.entity_buffer)
