Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

# Advanced Development with GXF / Graph Composer

[TOC]

## Using a GXF Release with CMake

A GXF release package generated with Bazel contains CMake rules to import binaries in a
consuming CMake project.

To use a GXF release package in a CMake project, simply add the following lines to `CMakeLists.txt`:

```
find_package(GXF 4.1 CONFIG REQUIRED)
```

Note that the GXF package does not distribute external library dependencies, which may be required to use
GXF in a consuming project. The consuming project must supply any library dependencies of the imported
GXF component, otherwise CMake will throw an error at configuration time. Dependencies with
CMake support may be provided by adding their path to [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/v3.26/variable/CMAKE_PREFIX_PATH.html) or supplying the path
as a configuration argument in the form `-Dmy_library_DIR=<path/to/library>`.

To minimize extraneous dependencies, you can specify only the GXF components you wish to import
as CMake targets from the GXF release package. Non-critical components that are not specified in
the component list will not be available to the CMake project.

```
find_package(GXF 4.1 CONFIG REQUIRED
    COMPONENTS
        core
        cuda
        logger
        multimedia
        std
)
```

After `find_package(GXF ...)` completes successfully, GXF CMake targets are available in the
CMake project for use in subsequent development.

```
add_library(MyLibrary myfile.cpp)
target_link_libraries(MyLibrary
    GXF::core
    GXF::std
)
```

Linking GXF targets to downstream libraries such as `MyLibrary` makes GXF public headers visible
for inclusion in C++ source and header files:

```cpp
#include <gxf/std/tensor.hpp>
```

Refer to CMake documentation for additional information:
- [CMake Key Concepts](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Key%20Concepts.html)
- [`find_package()`](https://cmake.org/cmake/help/v3.26/command/find_package.html)

## Building GXF with CMake

GXF supports CMake as a secondary build system to build, test, and package GXF core extensions
in local development. CMake build rules are _not_ currently employed for formal GXF releases.

### CMake Prerequisites

We recommend the following environment for building with CMake:
- A Linux environment (tested with Ubuntu 22.04);
- CMake >=3.24 (tested with CMake 3.27);
- Ninja build generator v1.11.1;
- gcc (tested with 11.4.0);
- CUDA Toolkit >= 12.x (tested with 12.3).

### CMake Build Steps

1. Create a working build directory outside of your source directory:
```
$ mkdir gxf-bld
$ cd gxf-bld
```

2. Configure the project with respect to the source directory with a presets and any custom options.
```
gxf-bld $ cmake --preset x86_64_cuda_12_2 -DBUILD_TESTING:BOOL=ON path/to/gxf
```

By default the project will run a "superbuild", which first collects and builds project
dependencies before building GXF in a subdirectory. You can find the "inner" build of
GXF in the `gxf-build` subdirectory. Alternatively, you can disable the dependency superbuild
with the configuration option `-DGXF_SUPERBUILD::BOOL=OFF` and pass in your own dependency
paths to CMake.

You must pass one of the following presets with `--preset <name>` to indicate the target configuration:
- `x86_64_cuda_12_2`
- `x86_64_cuda_12_6`
- `hp21ea_sbsa`
- `hp21ga_sbsa`
- `jetpack60`
- `jetpack61`

3. Build and install the project. By default the build will install output files
in the `gxf-install` build subdirectory.
```sh
gxf-bld $ cmake --build .
```

### Testing with CTest

To run tests with CTest, build the project with cmake, then run with CTest from the
inner GXF build directory:
```sh
gxf-bld $ cd gxf-build
gxf-bld/gxf-build $ ctest -C Release
```

To list all available tests:
```sh
gxf-bld/gxf-build $ ctest -C Release -N
```

Use the "-R" flag to filter for tests by name. Test names are derived from an inline
GoogleTest `TEST(Suite,Name)` function in the test source file or are set directly in the
CMake `add_test` function.
```sh
gxf-bld/gxf-build $ ctest -C Release -R MyTest
```

See [CTest documentation](https://cmake.org/cmake/help/v3.24/manual/ctest.1.html) for additional
command line options.

### Exporting with CMake

To use GXF libraries in an external project, build and install GXF with CMake,
then include GXF in your external project with `-DGXF_ROOT:PATH=path/to/gxf-bld/gxf-install`.

Alternatively, create an archive from the GXF installation directory, copy it to the target
system, and point to the installed GXF location.

```sh
gxf-bld/gxf-install $ tar cvzf gxf-cmake-installation.tar.gz ./*
```
