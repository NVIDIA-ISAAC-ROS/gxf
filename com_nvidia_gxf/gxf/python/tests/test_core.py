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
import unittest
import numpy as np

import gxf.core
import gxf.python.tests

APP_YAML = "gxf/python/tests/test_core.yaml"
MANIFEST_YAML = "gxf/python/tests/test_core_manifest.yaml"


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
        cids = gxf.core.component_find(
            context, eid, component_name="parameter_test")
        self.assertEqual(len(cids), 1)

        gxf.core.parameter_set_bool(context, cids[0], "fact", True)
        gxf.core.parameter_set_bool(context, cids[0], "rumor", False)
        gxf.core.parameter_set_int32(context, cids[0], "forty_two", 42)
        gxf.core.parameter_set_int32(context, cids[0], "minus_one", -1)
        gxf.core.parameter_set_str(context, cids[0], "some_text", "hello")
        more_text = "- a: st\n  b: ee\n- c: an\n  d: en\n- e:\n    - f: figy\n      g: g"
        gxf.core.parameter_set_str(context, cids[0], "more", more_text)

        self.assertEqual(gxf.core.parameter_get_bool(
            context, cids[0], "fact"), True)
        self.assertEqual(gxf.core.parameter_get_bool(
            context, cids[0], "rumor"), False)
        self.assertEqual(gxf.core.parameter_get_int32(
            context, cids[0], "forty_two"), 42)
        self.assertEqual(gxf.core.parameter_get_int32(
            context, cids[0], "minus_one"), -1)
        self.assertEqual(gxf.core.parameter_get_str(
            context, cids[0], "more"), more_text)

        gxf.core.graph_activate(context)
        eid = gxf.core.entity_find(context, "rx")
        self.assertEqual(gxf.core.entity_get_status(context, eid), 0)
        gxf.core.graph_run(context)
        self.assertEqual(gxf.core.entity_get_status(context, eid), 5)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_component_type_id(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_ping_py_cpp.yaml")
        tid = gxf.core.component_type_id(context, "nvidia::gxf::Transmitter")
        self.assertEqual(tid.hash1, int(0xc30cc60f0db2409d))
        self.assertEqual(tid.hash2, int(0x92b6b2db92e02cce))
        gxf.core.context_destroy(context)

    def test_python_ping_tx(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_ping_py_cpp.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_ping_rx(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_ping_cpp_py.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_ping_tx_rx(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_ping_py_py.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_cuda_dot_product(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_cuda_stream_dotproduct.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_data_on_device(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_data_on_device.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_stream_generator(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_cuda_stream_tensor_generator.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_data_on_device_on_host(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_cuda_stream_dotproduct_cupy.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_data_on_device_on_device(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_data_on_device_cupy.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_python_stream_generator_on_device(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_cuda_stream_tensor_generator_cupy.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_entity_deactivate(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_entity_delete.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_periodic_scheduling_term(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_periodic_scheduling_term.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_target_time_scheduling_term(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_target_time_scheduling_term.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)
        gxf.core.graph_deactivate(context)
        gxf.core.context_destroy(context)

    def test_graph_interrupt_incorrect_context(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_interrupt(0)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_CONTEXT_INVALID")

    def test_graph_interrupt_before_activate(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_load_file(
                context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
            gxf.core.graph_interrupt(context)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_INVALID_EXECUTION_SEQUENCE")

    def test_graph_interrupt_after_activate(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_load_file(
                context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
            gxf.core.graph_activate(context)
            gxf.core.graph_interrupt(context)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_INVALID_EXECUTION_SEQUENCE")

    def test_graph_interrupt_after_run(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_load_file(
                context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
            gxf.core.graph_activate(context)
            gxf.core.graph_run(context)
            gxf.core.graph_interrupt(context)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_INVALID_EXECUTION_SEQUENCE")

    def test_graph_interrupt_after_wait(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_load_file(
                context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
            gxf.core.graph_activate(context)
            gxf.core.graph_run_async(context)
            gxf.core.graph_wait(context)
            gxf.core.graph_interrupt(context)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_INVALID_EXECUTION_SEQUENCE")

    def test_graph_interrupt_after_deactivate(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_load_file(
                context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
            gxf.core.graph_activate(context)
            gxf.core.graph_run(context)
            gxf.core.graph_deactivate(context)
            gxf.core.graph_interrupt(context)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_INVALID_EXECUTION_SEQUENCE")

    def test_graph_consecutive_interrupt(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        try:
            gxf.core.graph_load_file(
                context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
            gxf.core.graph_activate(context)
            gxf.core.graph_run_async(context)
            gxf.core.graph_interrupt(context)
            gxf.core.graph_interrupt(context)
            assert(False)
        except ValueError as v:
            assert(v.__str__() == "GXF_INVALID_EXECUTION_SEQUENCE")

    def test_graph_interrupt(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_python_app_gxf_graph_wait.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run_async(context)
        gxf.core.graph_interrupt(context)


    def test_subgraph(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/test_subgraph.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run_async(context)
        gxf.core.graph_interrupt(context)

    def test_gxf_dtypes(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.int8))
               == "nvidia::gxf::PrimitiveType::kInt8")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.int16))
               == "nvidia::gxf::PrimitiveType::kInt16")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.int32))
               == "nvidia::gxf::PrimitiveType::kInt32")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.int64))
               == "nvidia::gxf::PrimitiveType::kInt64")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.uint8))
               == "nvidia::gxf::PrimitiveType::kUInt8")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.uint16))
               == "nvidia::gxf::PrimitiveType::kUInt16")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.uint32))
               == "nvidia::gxf::PrimitiveType::kUInt32")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.uint64))
               == "nvidia::gxf::PrimitiveType::kUInt64")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.float32))
               == "nvidia::gxf::PrimitiveType::kFloat32")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.float64))
               == "nvidia::gxf::PrimitiveType::kFloat64")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.complex64))
               == "nvidia::gxf::PrimitiveType::kCustom")
        assert(gxf.core.get_gxf_primitive_type(np.dtype(np.complex128))
               == "nvidia::gxf::PrimitiveType::kCustom")


if __name__ == '__main__':
    unittest.main()
