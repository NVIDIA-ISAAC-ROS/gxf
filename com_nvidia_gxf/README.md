Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

[TOC]

# GXF / Graph Composer

GXF is a modular and extensible framework to build high-performance AI applications.
- Enable developers to reuse components and app graphs between different products to build their own applications.
- Enable developers to use common data formats.
- Enable developers with tools to build and analyze their applications.

Visit the [GXF User Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/graphtools-docs/docs/text/GraphComposer_intro.html#gxf-graph-execution-format) to learn more.

# Building GXF

Developers may build GXF in one of three ways:
1. [Build with Bazel directly](#building-on-linux-host-with-bazel);
2. [Build with a Dazel container](#building-with-dazel) (recommended);
3. [Build with CMake](#building-with-cmake) (in development).

## Building on Linux Host - Ubuntu-22.04 with Bazel

Install dependencies required to setup your host environment
```
$ source gxf/engine/build/scripts/install_dependencies.sh
```

## Building on Linux Host - RHEL9 with Bazel

Install dependencies required to setup your host environment
```
$ source gxf/engine/build/scripts/install_dependencies_rhel.sh
```

The following host targets are supported:

| Bazel Config           |   Cuda     |    Cudnn      |
| :------------:         |   :----:   |   :---------: |
| x86_64_cuda_12_2       |   12.2     |    8.9.2      |
| x86_64_cuda_12_6       |   12.6     |    9.3.0      |
| x86_64_rhel9_cuda_12_2 |   12.2     |    8.9.2      |
| jetpack60              |   12.2     |    8.9.4      |
| jetpack61              |   12.6     |    9.3.0      |
| hp21ea_sbsa            |   12.2     |    8.9.2      |
| hp21ga_sbsa            |   12.6     |    9.3.0      |

Run the following command to build extensions for any of the above target platforms.

```
bazel build --config=<bazel config> ...
```

To cross compile aarch64 extensions for jetpack 6.0 on x86 host
```
bazel build --config=jetpack60 ...
```

Only ubuntu 22.04 OS is supported and support for ubuntu 20.04 has been deprecated. The default python version for ubuntu22.04 is python3.10.

### Controlling the Logging Level at Build Time

Define GXF_LOG_ACTIVE_LEVEL before including `common/logger.hpp` to control the logging level
at compile time. This allows you to skip logging at certain levels.

Example:

```cpp
#define GXF_LOG_ACTIVE_LEVEL 2
#include "common/logger.hpp"
...
```

With this setting, logging will occur only at the `WARNING(2)`, `ERROR(1)`, and `PANIC(0)` levels,
regardless of the runtime logging level (through `GXF_LOG_LEVEL` environment variable, `SetSeverity()` API, or `--severity` command line argument).

You can define `GXF_LOG_ACTIVE_LEVEL` in your build system.

In the Bazel build system, set this in your build configuration as follows:
```
cc_binary(
    name = "my_binary",
    srcs = ["my_binary.cc"],
    copts = ["-DGXF_LOG_ACTIVE_LEVEL=2"],
)
```

This sets the active logging level to `2` (WARNING) for the target `my_binary`.

Or, when using a Bazel build command:

```bash
bazel build --copt=-DGXF_LOG_ACTIVE_LEVEL=3 //path:to_your_target
```

This sets the active logging level to `INFO(3)` for the target `//path:to_your_target`.

By default, `GXF_LOG_ACTIVE_LEVEL` is set to `VERBOSE(5)`, which enables all logging levels.

## Building with Dazel

As an alternative of using Bazel on the host machine, we can leverage Dazel, which allows the execution of Bazel in a Docker container. GXF supports ubuntu 22.04 builds using Dazel. It can be installed on your host machine via:
```
$ pip3 install -U git+https://gitlab-master.nvidia.com/gxf/dazel.git
```
or
```
$ pip3 install -U git+ssh://git@gitlab-master.nvidia.com:12051/gxf/dazel.git
```
Docker version > 19.02 with nvidia-container-toolkit (also known as nvidia-docker) support is a prerequisite for using Dazel.

Install those components here:

    https://docs.docker.com/install/linux/docker-ce/ubuntu/
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Running "$dazel" will build a docker image with a GXF Dockerfile specified here. Use dazel similar to bazel within the GXF Repo. Commands, targets, flags are forwarded to the bazel process within the docker container.

For example,

    dazel test
    dazel run
    dazel build


Dazel will also mount certain host folders in order to properly interface with the GXF Registry (such as your local extension workspace). The user's UID and GID are used within the Docker container to avoid permission conflict.

Refer to the GXF [.dazelrc](https://git-master.nvidia.com/r/plugins/gitiles/gxf/gxf/+/refs/heads/master/.dazelrc) file for more insight as to what is mounted.

## Using with CMake

Refer to [advanced build instructions](DEVELOP.md).

## Running GXF applications

There are two ways to execute a graph application from the command line. The first one makes use of a predefined nv_gxf_app bazel target and the second one makes use of the GXE runtime binary.

To run a GXF application defined using a nv_gxf_app / nv_gxf_test_app targets, with bazel
```
bazel run //gxf/test/apps:test_ping
```

To run the same application within dazel,
```
dazel run //gxf/test/apps:test_ping
```

To same graph application can also be executed using the GXE runtime
```
bazel run //gxf/gxe:gxe -- -app gxf/test/apps/test_ping.yaml -manifest gxf/gxe/manifest.yaml
```

Cli args to GXE runtime can be also be specified. For example the log level is set using severity arg
```
bazel run //gxf/gxe:gxe -- -app gxf/test/apps/test_ping.yaml -manifest gxf/gxe/manifest.yaml  -severity=4
```

## Registering GXF Extensions

All extensions in this repository can be registered with
```
./register_extensions.sh
```
Variant for any specific platform can also registered using
```
./register_extensions.sh --config=jetpack60
```

At least one x86 variant must be registered first to generate the extension metadata before registering variant for any other platform

## Profiling GXF Applications

GXF core has nvtx hooks which help in profiling applications using nsight developer tools.

To enable the profiler while running the application, use the following arguments with the corresponding nv_gxf_app / nv_gxf_test_app targets.

```
dazel run //path/to:app_target -- --profile --export
```

To enable the profiler while running a gtest / cc_binary target, use the following arguments while running the executable
```
dazel run //path/to:exe_target --run_under="nsys profile --output=/tmp/test_entity"
```


The app run script generates a nsight system profile at /tmp/<app>.nsys-rep which can be opened the nsight-ui tool to analyse the profile visually.

## Debugging GXF Applications

Applications defined using nv_gxf_app / nv_gxf_test_app macro can be debugged by using the following arguments:

```
dazel run //path/to:app_target -- --[gdb|cuda_gdb|compute_sanitizer]
```

To enable a debugging tool like gdb while running a gtest / cc_binary targets, use the following arguments:

```
dazel run //path/to:exe_target --config=debug --run_under="/usr/bin/gdb --args"
```

## Deploying GXF applications on a remote machine

Bazel targets defined using nv_gxf_app / nv_gxf_test_app / nv_pygxf_app / nv_gxf_cc_app / nv_gxf_cc_test macro can be deployed remotely by the deploy script as follows:

```
./engine/build/scripts/deploy.sh -p //gxf/test/apps:test_or_scheduling_term-pkg -d jetpack60 -h 10.111.53.192 --remote_user ubuntu --deploy_path /tmp/test/ -b bazel
```

## Enabling pre-commit Git Hook

To enable, copy or symlink engine/build/scripts/pre-commit.sh into your .git/hooks folder:

To copy:

```
cp engine/build/scripts/pre-commit.sh .git/hooks/pre-commit # copy
```
To symlink:

```
ln -s `pwd`/engine/build/scripts/pre-commit.sh .git/hooks/pre-commit # symlink
```
## Package Validation

To verify the validity of your package (tarball_content.yaml), execute the following command:

```
bazel run //engine/build/scripts:run_release_package_validation
```
This command runs a comprehensive validation process on release package (tarball_content.yaml), ensuring all components are correctly configured and ready for deployment.

## Pre-Commit Checks

Before committing your changes, it's recommended to run the pre-commit checks. These checks include format validation, linting, and package verification. Use the following command:

```
bazel run //engine/build/scripts:pre_commit_check
```
Running this command helps maintain code quality and consistency across the project by catching potential issues early in the development process.