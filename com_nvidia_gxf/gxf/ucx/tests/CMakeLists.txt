# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

## Test Data

set(GXF_UCX_TEST_DATA
    test_forward_tx_rx.yaml
    test_forward_tx_rx_sync.yaml
    test_multiple_tensor_rx.yaml
    test_multiple_tensor_tx.yaml
    test_ping_ucx_rx.yaml
    test_ping_ucx_tx.yaml
    test_tensor_ucx_rx_cpu_only.yaml
    test_tensor_ucx_rx_EBS.yaml
    test_tensor_ucx_rx_EBS_sync.yaml
    test_tensor_ucx_rx_multiple_entity.yaml
    test_tensor_ucx_rx_multiple.yaml
    test_tensor_ucx_rx.yaml
    test_tensor_ucx_tx_EBS.yaml
    test_tensor_ucx_tx_EBS_sync.yaml
    test_tensor_ucx_tx_multiple1.yaml
    test_tensor_ucx_tx_multiple2.yaml
    test_tensor_ucx_tx_multiple.yaml
    test_tensor_ucx_tx.yaml
)

set(UCX_TEST_MANIFEST_FILEPATH "${CMAKE_CURRENT_BINARY_DIR}/manifest.yaml")
set(UCX_TEST_REQUIRED_EXTENSIONS "std;cuda;stream;test_extension;multimedia;serialization;ucx")
gxf_core_generate_manifest_file(EXTENSION_TARGETS "${UCX_TEST_REQUIRED_EXTENSIONS}")

set(GXF_UCX_TEST_DATA_OUTPUT "")
foreach(_file ${GXF_UCX_TEST_DATA})
    gxf_copy_to_output(${_file} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    add_test(
        NAME "Test_ucx_${_file}"
        COMMAND
            $<TARGET_FILE:gxe>
            -app ${CMAKE_CURRENT_BINARY_DIR}/${_file}
            -manifest ${UCX_TEST_MANIFEST_FILEPATH}
    )
    list(APPEND GXF_UCX_TEST_DATA_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_file}")
endforeach()
add_custom_target(copy_ucx_test_data ALL
       DEPENDS ${GXF_UCX_TEST_DATA_OUTPUT}
)

# TODO tests are listening on port, do they require multiple processes for IPC?
