'''
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
'''
import gxf.core
import gxf.std.vault_pybind
import unittest

APP_YAML = "gxf/python/tests/test_std_vault.yaml"
MANIFEST_YAML = "gxf/python/tests/test_std_vault_manifest.yaml"


class TestStdVault(unittest.TestCase):
    '''
    Test accessing messages stored in gxf::Vault component
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
        self.assertEqual(cid, 12)

        # Test with correct tid
        tid = gxf.core.tid_null()
        tid.hash1 = 1227454707155616515
        tid.hash2 = 13403506836782358117
        cids = gxf.core.component_find(context, eid, tid, "vault")
        self.assertEqual(len(cids), 1)
        self.assertEqual(cids[0], cid)
        # Test with wrong tid
        tid.hash1 = 12
        cids = gxf.core.component_find(context, eid, tid, "vault")
        self.assertEqual(len(cids), 0)

        num_steps = 10
        count_per_step = 100

        gxf.core.graph_activate(context)
        gxf.core.graph_run_async(context)

        for i in range(num_steps):
            entities = gxf.std.vault_pybind.store_blocking(context, cid, count_per_step)
            gxf.std.vault_pybind.free(context, cid, entities)

        gxf.core.graph_wait(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_basic_blocking_for_app(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(context, APP_YAML)

        eid = gxf.core.entity_find(context, "rx")
        cids = gxf.core.component_find(context, eid, component_name="vault")
        self.assertEqual(len(cids), 1)
        cid = cids[0]
        # self.assertEqual(cid, 11)

        # Test with correct tid
        tid = gxf.core.tid_null()
        tid.hash1 = 1227454707155616515
        tid.hash2 = 13403506836782358117
        cids = gxf.core.component_find(context, eid, tid, "vault")
        self.assertEqual(len(cids), 1)
        self.assertEqual(cids[0], cid)
        # Test with wrong tid
        tid.hash1 = 12
        cids = gxf.core.component_find(context, eid, tid, "vault")
        self.assertEqual(len(cids), 0)

        # 1 step will be from manual store_blocking
        num_steps = 9
        count_per_step = 100

        gxf.core.graph_activate(context)
        gxf.core.graph_run_async(context)

        # Initially block so to not store_blocking_for during start up
        entities = gxf.std.vault_pybind.store_blocking(context, cid, count_per_step)
        self.assertEqual(len(entities), count_per_step)
        gxf.std.vault_pybind.free(context, cid, entities)

        # Test lower duration than periodic term of 500000 (0.5 milliseconds) and multiply per
        # expected count
        short_duration_ns = 250000 * count_per_step
        entities = gxf.std.vault_pybind.store_blocking_for(context, cid, count_per_step, short_duration_ns)
        self.assertEqual(len(entities), 0)

        # Test longer duration than periodic term of 500000 (0.5 milliseconds) and multiply per
        # expected count.
        # Now should behave like store_blocking since notification for ready entities
        # should come prior to the timeout.
        long_duration_ns = 750000 * count_per_step
        for i in range(num_steps):
            # entities = gxf.StandardExtension.store_blocking(context, cid, count_per_step)
            entities = gxf.std.vault_pybind.store_blocking_for(context, cid, count_per_step, long_duration_ns)
            self.assertEqual(len(entities), count_per_step)
            gxf.std.vault_pybind.free(context, cid, entities)

        gxf.core.graph_wait(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)


if __name__ == '__main__':
    unittest.main()
