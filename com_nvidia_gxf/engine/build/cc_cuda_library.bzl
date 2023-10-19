'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

def cc_cuda_library(deps = [], copts = [], **kwargs):
    '''
    A custom library which supports building CUDA kernels. This is a thin wrapper around cc_library
    which adds a dependency on cuda. It also add "-x cuda" to deps to mark it for the compiler
    wrapper script. If this sees the "-x cuda" option it uses a special path around NVCC to compile
    the library.

    WARNING: As we are using the bazel C++ branch source files can not end with .cu or similar. We
    are choosing .cu.cpp and .cu.hpp as extensions for CUDA kernel code. This can be added to srcs
    or hdrs like normal C++ source or header files.

    WARNING: Beware that NVCC does not support all the features as the standard GCC, so some
    libraries might not compile yet when added as a dependency to a cc_cuda_library.
    '''

    native.cc_library(
        deps = deps + ["@com_nvidia_gxf//third_party:cuda"],
        copts = copts + ["-x cuda"],
        # TODO For some reason cudaMalloc and such are not present in the library
        # when cc_test targets are build. cc_test targets are linked dynamically.
        # The suggested fix is to put a linkstatic here to force static linking
        # for cc_test targets.
        linkstatic = True,
        **kwargs
    )
