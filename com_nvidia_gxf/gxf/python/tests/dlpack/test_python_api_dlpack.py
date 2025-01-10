"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import unittest

from MatMul import MatMul
from StepCount import StepCount
from TensorPairGenerator import TensorPairGenerator
from VerifyEqual import VerifyEqual

import gxf.std as std
from gxf.core import Graph
from gxf.python_codelet import PyCodeletV0
from gxf.std import (
    Broadcast,
    CountSchedulingTerm,
    DoubleBufferReceiver,
    DoubleBufferTransmitter,
    DownstreamReceptiveSchedulingTerm,
    Entity,
    GPUDevice,
    GreedyScheduler,
    MessageAvailableSchedulingTerm,
    MultiThreadScheduler,
    RealtimeClock,
)

# avoid potential permissions issue in dazel by not caching to disk
os.environ["CUPY_CACHE_IN_MEMORY"] = "1"


def add_tensor_generator_entity(
    graph,
    entity_name="TensorPairGenerator",
    codelet_name="tensor_pair_generator",
    use_array_interface=False,
    count=20,
):
    """Create an entity that generates pairs of host and device tensors."""
    ptx = graph.add(Entity(entity_name))
    ptx.add(DoubleBufferTransmitter(name="host_out"))
    ptx.add(
        DownstreamReceptiveSchedulingTerm(name="host_drst", transmitter=ptx.host_out, min_size=1)
    )
    ptx.add(DoubleBufferTransmitter(name="cuda_out"))
    ptx.add(
        DownstreamReceptiveSchedulingTerm(name="cuda_drst", transmitter=ptx.cuda_out, min_size=1)
    )
    ptx.add(
        PyCodeletV0(
            name=codelet_name,
            codelet=TensorPairGenerator,
            codelet_file="gxf/python/tests/dlpack/TensorPairGenerator.py",
            transmitter0=ptx.host_out,
            transmitter1=ptx.cuda_out,
            use_array_interface=use_array_interface,
            # could not pass shape tuple as parameter in Python API so use separate rows/cols arg
            rows=16,
            cols=64,
            # cannot pass numpy.dtype object as parameter in Python API, so use string representation
            dtype="f",
            # clock=clock,
        )
    )
    ptx.add(CountSchedulingTerm(name="cst", count=count))
    return ptx


def add_matmul_entity(
    graph,
    entity_name="MatMulHost",
    codelet_name="host_matmul",
    device="cpu",
    use_array_interface=False,
):
    pmm = graph.add(Entity(entity_name))
    pmm.add(DoubleBufferTransmitter(name="tx"))
    pmm.add(DownstreamReceptiveSchedulingTerm(name="drst", transmitter=pmm.tx, min_size=1))
    pmm.add(DoubleBufferReceiver(name="rx"))
    pmm.add(MessageAvailableSchedulingTerm(name="mast", receiver=pmm.rx, min_size=1))
    pmm.add(
        PyCodeletV0(
            name=codelet_name,
            codelet=MatMul,
            codelet_file="gxf/python/tests/dlpack/MatMul.py",
            rx=pmm.rx,
            tx=pmm.tx,
            device=device,
            use_array_interface=use_array_interface,
        )
    )
    return pmm


def add_verify_entity(
    graph,
    entity_name="VerifyEqual",
    codelet_name="verify_equal",
    use_array_interface=False,
    count=20,
):
    pverify = graph.add(Entity(entity_name))
    pverify.add(DoubleBufferReceiver(name="rx0"))
    pverify.add(MessageAvailableSchedulingTerm(name="mast0", receiver=pverify.rx0, min_size=1))
    pverify.add(DoubleBufferReceiver(name="rx1"))
    pverify.add(MessageAvailableSchedulingTerm(name="mast1", receiver=pverify.rx1, min_size=1))
    pverify.add(
        PyCodeletV0(
            name=codelet_name,
            codelet=VerifyEqual,
            codelet_file="gxf/python/tests/dlpack/VerifyEqual.py",
            receiver0=pverify.rx0,
            receiver1=pverify.rx1,
            use_array_interface=use_array_interface,
        )
    )
    pverify.add(
        PyCodeletV0(
            name="step_count",
            codelet=StepCount,
            codelet_file="gxf/python/tests/dlpack/StepCount.py",
            expected_count=count
        )
    )
    return pverify


class TestCore(unittest.TestCase):
    """Test application with Python codelets using array interoperability methods.

    This can be either via the DLPack Python protocol or via one of the array interfaces
    (__array_interface__ on host or __cuda_array_interface__ on device).
    """

    def run_app(
        self, scheduler="multithread", threads=6, use_array_interface=False, n_broadcast=1, count=20
    ):
        g = Graph()
        clock = std.set_clock(g, RealtimeClock(name="clock"))

        # create the tensor generator entity
        ptx = add_tensor_generator_entity(g, use_array_interface=use_array_interface, count=count)

        if n_broadcast < 2:
            # configure the host matrix multiply entity
            pmm_host = add_matmul_entity(
                g,
                entity_name="MatMulHost",
                codelet_name="host_matmul",
                device="cpu",
                use_array_interface=use_array_interface,
            )
            # configure the device matrix multiply entity
            pmm_cuda = add_matmul_entity(
                g,
                entity_name="MatMulCuda",
                codelet_name="cuda_matmul",
                device="gpu",
                use_array_interface=use_array_interface,
            )

            # configure the tensor verification entity
            pverify = add_verify_entity(
                g,
                entity_name="VerifyEqual",
                codelet_name="verify_equal",
                use_array_interface=use_array_interface,
                count=count,
            )

            std.connect(ptx.host_out, pmm_host.rx)
            std.connect(ptx.cuda_out, pmm_cuda.rx)
            std.connect(pmm_host.tx, pverify.rx0)
            std.connect(pmm_cuda.tx, pverify.rx1)
        else:
            # broadcast component for the host tensor port
            broadcast_host = g.add(Entity(name="broadcast_host"))
            broadcast_host.add(DoubleBufferReceiver(name="source"))
            broadcast_host.add(
                MessageAvailableSchedulingTerm(name="mast_host", receiver=broadcast_host.source, min_size=1)
            )
            for n in range(n_broadcast):
                broadcast_host.add(DoubleBufferTransmitter(name=f"host_out{n}"))
                broadcast_host.add(
                    DownstreamReceptiveSchedulingTerm(
                        name=f"host_drst{n}", transmitter=getattr(broadcast_host, f"host_out{n}"), min_size=1
                    )
                )
            broadcast_host.add(Broadcast(name="bcst_host", source=broadcast_host.source))

            # broadcast component for the cuda tensor port
            broadcast_cuda = g.add(Entity(name="broadcast_cuda"))
            broadcast_cuda.add(DoubleBufferReceiver(name="source"))
            broadcast_cuda.add(
                MessageAvailableSchedulingTerm(name="mast_cuda", receiver=broadcast_cuda.source, min_size=1)
            )
            for n in range(n_broadcast):
                broadcast_cuda.add(DoubleBufferTransmitter(name=f"cuda_out{n}"))
                broadcast_cuda.add(
                    DownstreamReceptiveSchedulingTerm(
                        name=f"cuda_drst{n}", transmitter=getattr(broadcast_cuda, f"cuda_out{n}"), min_size=1
                    )
                )
            broadcast_cuda.add(Broadcast(name="bcst_cuda", source=broadcast_cuda.source))

            std.connect(ptx.host_out, broadcast_host.source)
            std.connect(ptx.cuda_out, broadcast_cuda.source)

            for n in range(n_broadcast):
                host_mm = add_matmul_entity(
                    g,
                    entity_name=f"MatMulHost{n}",
                    codelet_name=f"host_matmul{n}",
                    device="cpu",
                    use_array_interface=use_array_interface,
                )
                cuda_mm = add_matmul_entity(
                    g,
                    entity_name=f"MatMulCuda{n}",
                    codelet_name=f"cuda_matmul{n}",
                    device="gpu",
                    use_array_interface=use_array_interface,
                )
                verify = add_verify_entity(
                    g,
                    entity_name=f"VerifyEqual{n}",
                    codelet_name=f"verify_equal{n}",
                    use_array_interface=use_array_interface,
                    count=count,
                )
                std.connect(getattr(broadcast_host, f"host_out{n}"), host_mm.rx)
                std.connect(getattr(broadcast_cuda, f"cuda_out{n}"), cuda_mm.rx)
                std.connect(host_mm.tx, verify.rx0)
                std.connect(cuda_mm.tx, verify.rx1)

        # add a GPUDevice for use by the default entity group
        g.add(Entity("GPU_0")).add(GPUDevice(name="GPU_0", dev_id=0))

        std.enable_job_statistics(g, clock=clock)

        if scheduler == "multithread":
            std.set_scheduler(
                g,
                MultiThreadScheduler(
                    clock=clock,
                    stop_on_deadlock=True,
                    stop_on_deadlock_timeout=10,
                    check_recession_period_ms=1,
                    worker_thread_number=threads,
                ),
            )
        elif scheduler == "greedy":
            std.set_scheduler(g, GreedyScheduler(max_duration_ms=1000000, clock=clock))
        else:
            raise ValueError(f"unsupported scheduler: {scheduler}")

        g.load_extensions()
        g.run_async()
        g.wait()
        g.destroy()

    def test_python_dlpack_greedy(self):
        self.run_app(scheduler="greedy", use_array_interface=False, n_broadcast=1)

    def test_python_dlpack_multithread(self):
        self.run_app(scheduler="multithread", threads=6, use_array_interface=False, n_broadcast=1)

    def test_python_dlpack_greedy_broadcast(self):
        self.run_app(scheduler="greedy", use_array_interface=False, n_broadcast=3)

    def test_python_dlpack_multithread_broadcast(self):
        self.run_app(scheduler="multithread", use_array_interface=False, n_broadcast=3)

    def test_python_array_interface_greedy(self):
        self.run_app(scheduler="greedy", use_array_interface=True, n_broadcast=1)

    def test_python_array_interface_multithread(self):
        self.run_app(scheduler="multithread", threads=6, use_array_interface=True, n_broadcast=1)

    def test_python_array_interface_greedy_broadcast(self):
        self.run_app(scheduler="greedy", use_array_interface=True, n_broadcast=3)

    def test_python_array_interface_multithread_broadcast(self):
        self.run_app(scheduler="multithread", use_array_interface=True, n_broadcast=3)


if __name__ == "__main__":
    unittest.main()
