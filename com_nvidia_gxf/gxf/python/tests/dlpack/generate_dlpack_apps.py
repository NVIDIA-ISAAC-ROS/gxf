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

import sys


"""
Generation of PyCodeletV0-based graphs.

The operators involved are:

1.) Tensor generator
  - emits a pair of host tensors on one port (host_out)
  - emits a pair of cuda tensors on another port (cuda_out)

2.) MatMul (matrix multiply)
  - receives a pair of tensors on the input port (rx)
  - emits a single tensor that is the result of matrix multiplication of those two tensors (tx)

3.) VerifyEqual
  - has two input ports:
      - rx0 receives from MatMul with device='cpu'
      - rx1 receives from MatMul with device='gpu'
  - checks that the host and device results are equal to within some tolerance.

If n_broadcast > 1:
  - There is also a 1:N broadcast of each of the tensor generator output ports
  - There are then N copies of the host and device MatMul and VerifyEqual codelets

Can be configured to use either greedy or multi-thread schedulers.

Can be configured to run for a given count and number of worker threads.

Can be configured to use either the DLPack interface or (CUDA) array interfaces.
"""


def create_generator(prefix1, prefix2, array_interface=False, count=200):
    code = f"""---
name: tensor_pair_generator
components:
- name: {prefix1}_out
  type: nvidia::gxf::DoubleBufferTransmitter
- name: {prefix2}_out
  type: nvidia::gxf::DoubleBufferTransmitter
- name: generator
  type: nvidia::gxf::PyCodeletV0
  parameters:
    codelet_name: TensorPairGenerator
    codelet_file: gxf/python/tests/dlpack/TensorPairGenerator.py
    codelet_params:
      transmitter0: {prefix1}_out
      transmitter1: {prefix2}_out
      use_array_interface: {array_interface}
      rows: 16
      cols: 64
      dtype: f
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: {prefix1}_out
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: {prefix2}_out
    min_size: 1
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: {count}
"""
    return code


def create_matmul(prefix, matmul_on_host, array_interface=False, n_broadcast=3):
    device = "cpu" if matmul_on_host else "gpu"
    if n_broadcast < 2:
        suffixes = ("",)
    else:
        suffixes = tuple(f"_{i}" for i in range(n_broadcast))

    code = ""
    for suffix in suffixes:
        code += f"""---
name: {prefix}_matmul{suffix}
components:
- name: rx
  type: nvidia::gxf::DoubleBufferReceiver
  parameters:
    capacity: 2
- name: tx
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx
    min_size: 1
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: tx
    min_size: 1
- type: nvidia::gxf::PyCodeletV0
  parameters:
    codelet_name: MatMul
    codelet_file: gxf/python/tests/dlpack/MatMul.py
    codelet_params:
      rx: rx
      tx: tx
      use_array_interface: {array_interface}
      device: {device}
"""
    return code


def create_broadcast(prefix, n_broadcast=3, count=200):
    if n_broadcast < 2:
        return ""

    code = f"""---
name: {prefix}_broadcast
components:
- name: source
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: source
    min_size: 1
"""

    for i in range(n_broadcast):
        code += f"""- name: {prefix}_{i}
  type: nvidia::gxf::DoubleBufferTransmitter
- type: nvidia::gxf::DownstreamReceptiveSchedulingTerm
  parameters:
    transmitter: {prefix}_{i}
    min_size: 1
"""
    code += f"""
- type: nvidia::gxf::Broadcast
  parameters:
    source: source
- type: nvidia::gxf::test::StepCount
  parameters:
    expected_count: {count}
"""
    return code


def create_verify(array_interface, n_broadcast=3, count=200):
    if n_broadcast < 2:
        suffixes = ("",)
    else:
        suffixes = tuple(f"_{i}" for i in range(n_broadcast))

    code = ""
    for suffix in suffixes:
        code += f"""---
name: verify_equal{suffix}
components:
- name: rx0
  type: nvidia::gxf::DoubleBufferReceiver
- name: rx1
  type: nvidia::gxf::DoubleBufferReceiver
- type: nvidia::gxf::PyCodeletV0
  parameters:
    codelet_name: VerifyEqual
    codelet_file: gxf/python/tests/dlpack/VerifyEqual.py
    codelet_params:
      receiver0: rx0
      receiver1: rx1
      use_array_interface: {array_interface}
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx0
    min_size: 1
- type: nvidia::gxf::MessageAvailableSchedulingTerm
  parameters:
    receiver: rx1
    min_size: 1
- name: step_count
  type: nvidia::gxf::PyCodeletV0
  parameters:
    codelet_name: StepCount
    codelet_file: gxf/python/tests/dlpack/StepCount.py
    codelet_params:
      expected_count: {count}
"""
    return code


def create_broadcast_connection(prefix, n_broadcast=3):
    if n_broadcast < 2:
        return ""

    # add connection to broadcast input
    code = f"""---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tensor_pair_generator/{prefix}_out
    target: {prefix}_broadcast/source
"""
    return code


def create_matmul_connections(prefix, n_broadcast=3):
    if n_broadcast < 2:
        suffixes = ("",)
        code = f"""---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: tensor_pair_generator/{prefix}_out
    target: {prefix}_matmul/rx
"""
    else:
        suffixes = tuple(f"_{i}" for i in range(n_broadcast))

        # add connections from broadcast output
        code = ""
        for suffix in suffixes:
            code += f"""---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: {prefix}_broadcast/{prefix}{suffix}
    target: {prefix}_matmul{suffix}/rx
"""
    return code


def create_verify_connections(prefix1, prefix2, n_broadcast=3):
    if n_broadcast < 2:
        suffixes = ("",)
    else:
        suffixes = tuple(f"_{i}" for i in range(n_broadcast))

    code = ""
    for suffix in suffixes:
        code += f"""---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: {prefix1}_matmul{suffix}/tx
    target: verify_equal{suffix}/rx0
---
components:
- type: nvidia::gxf::Connection
  parameters:
    source: {prefix2}_matmul{suffix}/tx
    target: verify_equal{suffix}/rx1
"""
    return code


def create_app(
    host_only=False,
    array_interface=False,
    scheduler="greedy",
    n_broadcast=0,
    count=200,
    workers=1,
):
    header = f"""%YAML 1.2
---
dependencies:
- extension: StandardExtension
  uuid: 8ec2d5d6-b5df-48bf-8dee-0252606fdd7e
  version: 2.6.0
- extension: PythonCodeletExtension
  uuid: 787daddc-1c34-11ec-9621-0242ac130002
  version: 0.6.0
"""

    if host_only:
        prefix1 = "host1"
        prefix2 = "host2"
        matmul1_on_host = True
        matmul2_on_host = True
    else:
        prefix1 = "host"
        prefix2 = "cuda"
        matmul1_on_host = True
        matmul2_on_host = False

    # create tensor generation codelet
    codelets = create_generator(
        prefix1, prefix2, array_interface=array_interface, count=count
    )

    if n_broadcast > 1:
        # create broadcast codelets
        codelets += create_broadcast(prefix1, n_broadcast=n_broadcast, count=count)
        codelets += create_broadcast(prefix2, n_broadcast=n_broadcast, count=count)

    # create matrix multiplication codelets
    codelets += create_matmul(
        prefix1,
        matmul1_on_host,
        array_interface=array_interface,
        n_broadcast=n_broadcast,
    )
    codelets += create_matmul(
        prefix2,
        matmul2_on_host,
        array_interface=array_interface,
        n_broadcast=n_broadcast,
    )

    # create verification codelet(s)
    both_on_host = matmul1_on_host and matmul2_on_host
    codelets += create_verify(
        array_interface=array_interface,
        n_broadcast=n_broadcast,
        count=count,
    )

    connections = ""
    if n_broadcast > 1:
        connections += create_broadcast_connection(prefix1, n_broadcast)
        connections += create_broadcast_connection(prefix2, n_broadcast)
    connections += create_matmul_connections(prefix1, n_broadcast)
    connections += create_matmul_connections(prefix2, n_broadcast)
    connections += create_verify_connections(prefix1, prefix2, n_broadcast)

    if scheduler == "greedy":
        scheduler = """---
name: scheduler
components:
- name: clock
  type: nvidia::gxf::RealtimeClock
- type: nvidia::gxf::GreedyScheduler
  parameters:
    clock: clock
    max_duration_ms: 10000
"""
    elif scheduler == "multithread":
        scheduler = f"""---
name: scheduler
components:
- name: clock
  type: nvidia::gxf::RealtimeClock
- type: nvidia::gxf::MultiThreadScheduler
  parameters:
    clock: clock
    max_duration_ms: 10000
    stop_on_deadlock: true
    stop_on_deadlock_timeout: 10
    check_recession_period_ms: 1
    worker_thread_number: {workers}
"""

    resources = """# Resources
# GPU
---
# only the first will be used by default EntityGroup
name: GPU_0
components:
- type: nvidia::gxf::GPUDevice
  name: GPU_0
  parameters:
    dev_id: 0

"""
    return header + codelets + connections + scheduler + resources


def generate_yaml_files(out_dir="/tmp"):
    count = 20

    # cases involving the DLPack interface
    app = create_app(
        array_interface=False, scheduler="greedy", count=count, n_broadcast=0
    )
    save(out_dir + "/test_tensor_dlpack_greedy.yaml", app)

    app = create_app(
        array_interface=False, scheduler="greedy", count=count, n_broadcast=3
    )
    save(out_dir + "/test_tensor_dlpack_greedy_broadcast3.yaml", app)

    app = create_app(
        array_interface=False, scheduler="multithread", count=count, n_broadcast=0, workers=4,
    )
    save(out_dir + "/test_tensor_dlpack_multithread.yaml", app)

    app = create_app(
        array_interface=False, scheduler="multithread", count=count, n_broadcast=3, workers=4,
    )
    save(out_dir + "/test_tensor_dlpack_multithread_broadcast3.yaml", app)

    app = create_app(
        array_interface=False, scheduler="multithread", count=count, n_broadcast=16, workers=8,
    )
    save(out_dir + "/test_tensor_dlpack_multithread_broadcast16.yaml", app)

    # cases involving __array_interface__ and __cuda_array_interface__
    app = create_app(
        array_interface=True, scheduler="greedy", count=count, n_broadcast=3,
    )
    save(out_dir + "/test_tensor_dlpack_greedy_broadcast3_interface.yaml", app)

    app = create_app(
        array_interface=True, scheduler="multithread", count=count, n_broadcast=3, workers=6,
    )
    save(out_dir + "/test_tensor_dlpack_multithread_broadcast3_interface.yaml", app)


def save(filename, text):
    with open(filename, "w") as file:
        file.write(text)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    generate_yaml_files(out_dir)
