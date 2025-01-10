'''
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
'''
from .allocator_pybind import *    # pylint: disable=no-name-in-module
from .clock_pybind import *    # pylint: disable=no-name-in-module
from .eos_pybind import *    # pylint: disable=no-name-in-module
from .receiver_pybind import *    # pylint: disable=no-name-in-module
from .transmitter_pybind import *    # pylint: disable=no-name-in-module
from .tensor_pybind import *     # pylint: disable=no-name-in-module
from .vault_pybind import *     # pylint: disable=no-name-in-module
from .timestamp_pybind import *     # pylint: disable=no-name-in-module
from .scheduling_terms_pybind import *     # pylint: disable=no-name-in-module
from .scheduling_condition_pybind import *     # pylint: disable=no-name-in-module
from .standard_graph import *
from .compute_entity import *
from .standard_methods import *
from .standard_methods import _generate_name

from gxf.core import Entity
from gxf.core import Component

try:
    from .Components import *
except:
    pass

from gxf.std.tensor_pybind import _PyLazyDLManagedTensorDeleter

_lazy_deleter = _PyLazyDLManagedTensorDeleter()
