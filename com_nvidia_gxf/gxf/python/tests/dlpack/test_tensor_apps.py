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
import unittest

import numpy as np
from generate_dlpack_apps import generate_yaml_files

import gxf.core
import gxf.python.tests

MANIFEST_YAML = "gxf/python/tests/dlpack/test_tensor_apps_manifest.yaml"

# generate the various YAML files for the test cases in TestCore
generate_yaml_files(out_dir="gxf/python/tests/dlpack/")


class TestTensorInteropGreedy(unittest.TestCase):
    """Test DLPack interface with NumPy and CuPy-based operators."""

    def test_tensor_greedy(self):
        """Linear pipeline with greedy scheduler."""
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_greedy.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)

    def test_tensor_greedy_broadcast(self):
        """Parallel pipeline (via broadcast) with greedy scheduler"""
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_greedy_broadcast3.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)

    def test_tensor_greedy_array_interface(self):
        """Linear pipeline with greedy scheduler."""
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_greedy_broadcast3_interface.yaml"
        )
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)


class TestTensorInteropMultithread(unittest.TestCase):
    """Test DLPack interface with NumPy and CuPy-based operators."""

    def test_tensor_multithread(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_multithread.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)

    def test_tensor_multithread_broadcast(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_multithread_broadcast3.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)

    def test_tensor_multithread_broadcast16(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_multithread_broadcast16.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)

    def test_tensor_multithread_array_interface(self):
        context = gxf.core.context_create()
        self.assertIsNotNone(context)
        gxf.core.load_extensions(context, manifest_filenames=[MANIFEST_YAML])
        gxf.core.graph_load_file(
            context, "gxf/python/tests/dlpack/test_tensor_dlpack_multithread_broadcast3_interface.yaml")
        gxf.core.graph_activate(context)
        gxf.core.graph_run(context)


if __name__ == "__main__":
    unittest.main()
