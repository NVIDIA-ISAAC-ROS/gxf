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
import os
import unittest

import cupy as cp
import numpy as np

import gxf.core
from gxf.std import tensor_pybind
from gxf.std import timestamp_pybind  # noqa: F401
from gxf.std import vault_pybind  # noqa: F401

APP_YAML = "gxf/python/tests/test_std_tensor.yaml"
MANIFEST_YAML = "gxf/python/tests/test_std_tensor_manifest.yaml"

# don't cache kernels to disk (avoids CI failure due to lack of write permissions)
os.environ["CUPY_CACHE_IN_MEMORY"] = "1"


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
            tensor = gxf.std.tensor_pybind.get_from_entity_context(context, entities[0], "tensor")
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

    def testFromAsTensorMethod(self):
        x = np.ones((16,), order='C')
        t = tensor_pybind.Tensor.as_tensor(x)
        assert isinstance(t, tensor_pybind.Tensor)
        assert isinstance(t, tensor_pybind.PyTensor)

        # as_tensor also adds the __array_interface__ dict for CPU arrays
        assert hasattr(t, '__array_interface__')
        # Check that values match, but it is not the same object
        # Note: The following comparison will fail for order='C' arrays because
        # the NumPy case will have __array_interface__['strides'] == None, but
        # PyTensor always sets the strides explicitly.
        # assert t.__array_interface__ == x.__array_interface__
        assert t.__array_interface__ is not x.__array_interface__

        y = np.asarray(t)
        np.testing.assert_allclose(x, y)

    def testTensorConstructorCuPyException(self):
        x_gpu = cp.empty((16,), order='C')

        # constructor with unsupported array type raises ValueError
        with self.assertRaises(ValueError) as context:
            tensor_pybind.Tensor(x_gpu)

        # error message mentions Tensor.as_tensor
        self.assertTrue("Tensor.as_tensor" in str(context.exception))

    def testTensorConstructorListException(self):
        # constructor with unsupported Python object type raises an exception
        with self.assertRaises(ValueError) as context:
            tensor_pybind.Tensor([1, 2, 3, 4, 5])

        # error message mentions Tensor.as_tensor
        self.assertTrue(
            "constructor can only create a tensor from a NumPy array" in str(context.exception)
        )

    def testTensorConstructorNumPy(self):
        # Tensor(x_host) will work (but GXF_LOG_WARNING is called)
        x_host = np.empty((16,), order='C')
        t = tensor_pybind.Tensor(x_host)
        np.testing.assert_array_equal(x_host, np.asarray(t))

    def test_tensor_from_dlpack_roundtrip(self):
        x = np.ones((16,), order='C')
        t = tensor_pybind.Tensor.from_dlpack(x)
        y = np.from_dlpack(t)
        np.testing.assert_allclose(x, y)

    def test_float16_tensor_from_dlpack_roundtrip(self):
        x = np.arange(16, dtype=np.float16)
        t = tensor_pybind.Tensor.from_dlpack(x)
        y = np.from_dlpack(t)
        assert y.dtype == np.float16
        np.testing.assert_allclose(x, y)

    def test_tensor_via_array_interface_roundtrip(self):
        x = np.ones((16,), order='C')
        t = tensor_pybind.Tensor.as_tensor(x)
        y = np.asarray(t)
        np.testing.assert_allclose(x, y)

    def test_float16_tensor_via_array_interface_roundtrip(self):
        x = np.arange(16, dtype=np.float16)
        t = tensor_pybind.Tensor.as_tensor(x)
        y = np.from_dlpack(t)
        assert y.dtype == np.float16
        np.testing.assert_allclose(x, y)

    def test_float16_tensor_via_cuda_array_interface_roundtrip(self):
        x = cp.arange(16, dtype=cp.float16)
        t = tensor_pybind.Tensor.as_tensor(x)
        y = cp.from_dlpack(t)
        assert y.dtype == cp.float16
        cp.testing.assert_allclose(x, y)

    def test_tensor_via_as_tensor_then_from_dlpack(self):
        x = np.ones((16,), order='C')
        t = tensor_pybind.Tensor.as_tensor(x)
        y = np.from_dlpack(t)
        np.testing.assert_allclose(x, y)

    def test_tensor_via_from_dlpack_then_asarray(self):
        x = np.ones((16,), order='C')
        t = tensor_pybind.Tensor.from_dlpack(x)
        y = np.asarray(t)
        np.testing.assert_allclose(x, y)


if __name__ == '__main__':
    unittest.main()
