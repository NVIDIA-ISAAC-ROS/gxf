'''
 SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gxf.core
from  gxf.std import vault_pybind
from gxf.std import tensor_pybind
from gxf.std import timestamp_pybind
import numpy as np
import unittest

APP_YAML = "gxf/python/tests/test_std_tensor.yaml"
MANIFEST_YAML = "gxf/python/tests/test_std_tensor_manifest.yaml"


class TestTensorAccess(unittest.TestCase):
    '''
    Test accessing tensor buffer and timestamp stored in gxf::Vault component
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

    def test_basic_app(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(context, APP_YAML)

        eid = gxf.core.entity_find(context, "rx")
        cids = gxf.core.component_find(context, eid, component_name="vault")
        self.assertEqual(len(cids), 1)
        cid = cids[0]

        num_steps = 10
        count_per_step = 100

        gxf.core.graph_activate(context)
        gxf.core.graph_run_async(context)

        for i in range(num_steps):
            entities = gxf.std.vault_pybind.store_blocking(context, cid, count_per_step)
            tensor = gxf.std.tensor_pybind.as_tensor(context, entities[0], "tensor")
            acq_time, pub_time = gxf.std.timestamp_pybind.as_timestamp(context, entities[0], "timestamp")
            self.assertIsNotNone(tensor)
            self.assertIsNotNone(acq_time)
            self.assertIsNotNone(pub_time)
            print(tensor)
            print(acq_time)
            print(pub_time)
            self.assertEqual(len(entities), count_per_step)
            self.assertEqual(tensor.shape[0], 2)
            self.assertEqual(tensor.shape[1], 2)
            gxf.std.vault_pybind.free(context, cid, entities)

        gxf.core.graph_wait(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)


if __name__ == '__main__':
    unittest.main()
