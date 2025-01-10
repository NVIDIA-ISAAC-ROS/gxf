"""
 SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gxf.core
import unittest

APP_YAML = "gxf/python/tests/test_tensor_info.yaml"
MANIFEST_YAML = "gxf/python/tests/test_tensor_info_manifest.yaml"


class TestTensorInfo(unittest.TestCase):
    '''
    Testing tensor info API
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
        gxf.core.graph_load_file(
            context, APP_YAML)
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

if __name__ == '__main__':
    unittest.main()