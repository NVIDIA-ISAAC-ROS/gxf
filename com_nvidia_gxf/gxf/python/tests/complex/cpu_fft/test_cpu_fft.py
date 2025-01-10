"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MANIFEST_YAML = "gxf/python/tests/complex/cpu_fft/test_cpu_fft_manifest.yaml"
APP_YAML = "gxf/python/tests/complex/cpu_fft/test_cpu_fft.yaml"

# the following YAML file will be generated while running the tests
DLPACK_APP_YAML = APP_YAML.replace(".yaml", "_dlpack.yaml")


class TestCPUfft(unittest.TestCase):
    '''
    Test creating and consuming a complex valued tensor in python
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
        # gxf.core.gxf_set_severity(context, 5)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(context, APP_YAML)
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_basic_app_dlpack(self):
        with open(APP_YAML, 'rt') as in_file, open(DLPACK_APP_YAML, 'wt') as out_file:
            yaml = in_file.read()
            yaml = yaml.replace("use_dlpack: 0", "use_dlpack: 1")
            # print(yaml)
            out_file.write(yaml)

        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        # gxf.core.gxf_set_severity(context, 5)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(context, DLPACK_APP_YAML)
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

if __name__ == '__main__':
    unittest.main()
