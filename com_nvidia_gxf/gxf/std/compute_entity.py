'''
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
'''

from gxf.core import Entity
from gxf.core import Component

try:
    from .Components import *
except:
    pass


class ComputeEntity(Entity):
    """Compute Entity adds an entity in the graph with basic scheduling terms.
    A codelet can be added to insert custom user code
    """

    def __init__(self, name: str = "", count: int = -1,
                 recess_period: int = 0, **kwargs):
        super().__init__(name, True)
        self._recess_period = recess_period
        self._count = count
        self._kwargs = kwargs
        if recess_period:
            self.add(PeriodicSchedulingTerm("pst",
                                            recess_period=recess_period))
        if count >= 0:
            self.add(CountSchedulingTerm("cst",
                                         count=count))

    def add_codelet(self, codelet, min_message_available=1, min_message_reception=1, rx_capacity: int = 1, tx_capacity: int = 1, tx_num: int = 1, rx_num: int = 1) -> Component:
        for _, info in codelet._validation_info_parameters.items():
            key = info['key']
            match info['handle_type']:
                case "nvidia::gxf::Transmitter":
                    rank: int = info['rank']
                    shape: list = info['shape']
                    if self._is_scalar(rank):
                        self._add_single_tx(codelet, key, tx_capacity, min_message_reception)
                    elif self._is_1D_array_any_len(rank, shape):
                        self._add_tx_list(codelet, key, tx_capacity, min_message_reception, tx_num)
                    elif self._is_1D_array_fix_len(rank, shape):
                        self._add_tx_list(codelet, key, tx_capacity, min_message_reception, shape[0])
                    elif self._is_2D_or_high_dim_array(rank):
                        shape_str = ', '.join(map(str, shape))
                        param_info = f"key: {key}, rank: {rank}, shape: {shape_str}"
                        raise RuntimeError("Unsupported rank and shape combination for codelet Transmitter parameter: " + param_info)
                case "nvidia::gxf::Receiver":
                    rank: int = info['rank']
                    shape: list = info['shape']
                    if self._is_scalar(rank):
                        self._add_single_rx(codelet, key, rx_capacity, min_message_available)
                    elif self._is_1D_array_any_len(rank, shape):
                        self._add_rx_list(codelet, key, rx_capacity, min_message_available, rx_num)
                    elif self._is_1D_array_fix_len(rank, shape):
                        self._add_rx_list(codelet, key, rx_capacity, min_message_available, shape[0])
                    elif _is_2D_or_high_dim_array(rank):
                        shape_str = ', '.join(map(str, shape))
                        param_info = f"key: {key}, rank: {rank}, shape: {shape_str}"
                        raise RuntimeError("Unsupported rank and shape combination for codelet Receiver parameter: " + param_info)
        self.add(codelet)
        return codelet

    def _add_single_tx(self, codelet, codelet_tx_param_key: str, tx_capacity: int, min_message_reception: int):
        tx_name = codelet_tx_param_key
        self.add(DoubleBufferTransmitter(name=tx_name, capacity=tx_capacity))
        self.add(DownstreamReceptiveSchedulingTerm(name='drst',
                                                transmitter=getattr(self, tx_name),
                                                min_size=min_message_reception))
        # set the transmitter of the codelet since we are not passing the
        # params anymore
        codelet._params[codelet_tx_param_key] = getattr(self, tx_name)
        setattr(codelet, codelet_tx_param_key, getattr(self, tx_name))

    def _add_single_rx(self, codelet, codelet_rx_param_key: str, rx_capacity: int, min_message_available: int):
        rx_name = codelet_rx_param_key
        self.add(DoubleBufferReceiver(name=rx_name, capacity=rx_capacity))
        self.add(MessageAvailableSchedulingTerm(name='mast',
                                                receiver=getattr(self, rx_name),
                                                min_size=min_message_available))
        # set the receiver of the codelet since we are not passing the
        # params anymore
        codelet._params[codelet_rx_param_key] = getattr(self, rx_name)
        setattr(codelet, codelet_rx_param_key, getattr(self, rx_name))

    def _add_tx_list(self, codelet, codelet_tx_list_param_key: str, tx_capacity: int, min_message_reception: int, tx_num: int):
        tx_list = []
        for i in range(tx_num):
            tx_i_name = codelet_tx_list_param_key + "_" + str(i)
            self.add(DoubleBufferTransmitter(name=tx_i_name, capacity=tx_capacity))
            self.add(DownstreamReceptiveSchedulingTerm(name=f'drst_{tx_i_name}',
                                                    transmitter=getattr(self, tx_i_name),
                                                    min_size=min_message_reception))
            tx_list.append(getattr(self, tx_i_name))
        codelet._params[codelet_tx_list_param_key] = tx_list
        # for connection, users address certain tx object by index from a list
        setattr(codelet, codelet_tx_list_param_key, tx_list)
        # we have to add the list to entity if we want keep consistency on g.{ENTITY_NAME}.{CODELET_PARAM_NAME}
        setattr(self, codelet_tx_list_param_key, tx_list)

    def _add_rx_list(self, codelet, codelet_rx_list_param_key: str, rx_capacity: int, min_message_available: int, rx_num: int):
        rx_list = []
        for i in range(rx_num):
            rx_i_name = codelet_rx_list_param_key + "_" + str(i)
            self.add(DoubleBufferReceiver(name=rx_i_name, capacity=rx_capacity))
            self.add(MessageAvailableSchedulingTerm(name=f'mast_{rx_i_name}',
                                                receiver=getattr(self, rx_i_name),
                                                min_size=min_message_available))
            rx_list.append(getattr(self, rx_i_name))
        codelet._params[codelet_rx_list_param_key] = rx_list
        # for connection, users address certain rx object by index from a list
        setattr(codelet, codelet_rx_list_param_key, rx_list)
        # we have to add the list to entity if we want keep consistency on g.{ENTITY_NAME}.{CODELET_PARAM_NAME}
        setattr(self, codelet_rx_list_param_key, rx_list)


    def _is_scalar(self, rank: int, shape: list = None):
        return rank == 0

    def _is_1D_array_any_len(self, rank: int, shape: list):
        check_rank: bool = rank == 1
        check_shape: bool = False
        if shape:  # if not empty
            # shape[0] 0, -1, -2, ...
            check_shape: bool = len(shape) == 1 and shape[0] < 1
        return check_rank and check_shape

    def _is_1D_array_fix_len(self, rank: int, shape: list):
        check_rank: bool = rank == 1
        check_shape: bool = False
        if shape:  # if not empty
            # shape[0] 1, 2, 3, ...
            check_shape: bool = len(shape) == 1 and shape[0] > 0
        return check_rank and check_shape

    def _is_2D_or_high_dim_array(self, rank: int, shape: list = None):
        return rank > 1
