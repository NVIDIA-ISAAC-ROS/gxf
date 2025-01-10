"""
 SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

def cc_cuda_library(deps = [], copts = [], srcs = [], add_cpp_extensions = False, **kwargs):
    '''
    A custom library which supports building CUDA kernels. This is a thin wrapper around cc_library
    which adds a dependency on cuda. It also add "-x cuda" to deps to mark it for the compiler
    wrapper script. If this sees the "-x cuda" option it uses a special path around NVCC to compile
    the library.

    WARNING: As we are using the bazel C++ branch source files can not end with .cu or similar. We
    are choosing .cu.cpp and .cu.hpp as extensions for CUDA kernel code. This can be added to srcs
    or hdrs like normal C++ source or header files. For external code that comes with .cu extensions,
    the boolean add_cpp_extensions can be enabled to create renamed copies with cu.cpp extensions.

    WARNING: Beware that NVCC does not support all the features as the standard GCC, so some
    libraries might not compile yet when added as a dependency to a cc_cuda_library.
    '''

    native.cc_library(
        deps = deps + ["@com_nvidia_gxf//third_party:cuda"],
        copts = copts + ["-x cuda"],
        srcs = _add_cpp_extensions(srcs) if add_cpp_extensions else srcs,
        # TODO For some reason cudaMalloc and such are not present in the library
        # when cc_test targets are build. cc_test targets are linked dynamically.
        # The suggested fix is to put a linkstatic here to force static linking
        # for cc_test targets.
        linkstatic = True,
        **kwargs
    )

def _add_cpp_extensions(srcs):
    """
    For all .cu files in src, create copies that has .cu.cpp extensions.
    A list containing all modified + unmodified paths is returned.
    """
    output = []

    for src in srcs:
        if src.endswith(".cu"):
            src_cpp = src.replace(".cu", ".cu.cpp")
            native.genrule(
                name = src + "_cu_to_cucpp",
                srcs = [src],
                outs = [src_cpp],
                cmd = "cp $(SRCS) $(OUTS)",
            )
            output.append(src_cpp)
        else:
            output.append(src)

    return output
