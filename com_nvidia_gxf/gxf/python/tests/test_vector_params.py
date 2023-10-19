'''
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

APP_YAML = "gxf/python/tests/test_vector_params.yaml"
MANIFEST_YAML = "gxf/python/tests/test_vector_params_manifest.yaml"


class TestCore(unittest.TestCase):
    '''
    Test loading subgraph via the application API
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
        eid = gxf.core.entity_find(context, "parameter_test")
        cids = gxf.core.component_find(context, eid, component_name="parameter_test")
        self.assertEqual(len(cids), 1)

        set_1d_float64_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
        gxf.core.parameter_set_1d_float64_vector(
            context, cids[0], "five_64_bit_floats", set_1d_float64_vector, 5)
        get_1d_float64_vector = gxf.core.parameter_get_1d_float64_vector(
            context, cids[0], "five_64_bit_floats", 5)
        assert(set_1d_float64_vector == get_1d_float64_vector)

        set_1d_int64_vector = [2, -4, 6, -8, 10]
        gxf.core.parameter_set_1d_int64_vector(
            context, cids[0], "five_64_bit_ints", set_1d_int64_vector, 5)
        get_1d_int64_vector = gxf.core.parameter_get_1d_int64_vector(
            context, cids[0], "five_64_bit_ints", 5)
        assert(set_1d_int64_vector == get_1d_int64_vector)

        set_1d_uint64_vector = [1, 1, 2, 3, 5]
        gxf.core.parameter_set_1d_uint64_vector(
            context, cids[0], "five_64_bit_unsigned_ints", set_1d_uint64_vector, 5)
        get_1d_uint64_vector = gxf.core.parameter_get_1d_uint64_vector(
            context, cids[0], "five_64_bit_unsigned_ints", 5)
        assert(set_1d_uint64_vector == get_1d_uint64_vector)

        set_1d_int32_vector = [-1, 5, -25, 125, -625]
        gxf.core.parameter_set_1d_int32_vector(
            context, cids[0], "five_32_bit_ints", set_1d_int32_vector, 5)
        get_1d_int32_vector = gxf.core.parameter_get_1d_int32_vector(
            context, cids[0], "five_32_bit_ints", 5)
        assert(set_1d_int32_vector == get_1d_int32_vector)

        set_2d_float_vector = [[1.0, 2, 3], [4, 5, 6]]
        gxf.core.parameter_set_2d_float64_vector(
            context, cids[0], "six_64_bit_float_2d", set_2d_float_vector, 2, 3)
        get_2d_float_vector = gxf.core.parameter_get_2d_float64_vector(
            context, cids[0], "six_64_bit_float_2d", 2, 3)
        assert(set_2d_float_vector == get_2d_float_vector)

        set_2d_int64_vector = [[1, 2, 3], [4, 5, 6]]
        gxf.core.parameter_set_2d_int64_vector(
            context, cids[0], "six_64_bit_int_2d", set_2d_int64_vector, 2, 3)
        get_2d_int64_vector = gxf.core.parameter_get_2d_int64_vector(
            context, cids[0], "six_64_bit_int_2d", 2, 3)
        assert(set_2d_int64_vector == get_2d_int64_vector)

        set_2d_uint64_vector = [[1, 2, 3], [4, 5, 6]]
        gxf.core.parameter_set_2d_uint64_vector(
            context, cids[0], "six_64_bit_uint_2d", set_2d_uint64_vector, 2, 3)
        get_2d_uint64_vector = gxf.core.parameter_get_2d_uint64_vector(
            context, cids[0], "six_64_bit_uint_2d", 2, 3)
        assert(set_2d_uint64_vector == get_2d_uint64_vector)

        set_2d_int32_vector = [[-1, 2, -3], [-4, -5, 6]]
        gxf.core.parameter_set_2d_int32_vector(
            context, cids[0], "six_32_bit_int_2d", set_2d_int32_vector, 2, 3)
        get_2d_int32_vector = gxf.core.parameter_get_2d_int32_vector(
            context, cids[0], "six_32_bit_int_2d", 2, 3)
        assert(set_2d_int32_vector == get_2d_int32_vector)

        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)


if __name__ == '__main__':
    unittest.main()
