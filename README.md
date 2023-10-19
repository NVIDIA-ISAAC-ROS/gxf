# Graph eXecution Framework (GXF)

## Overview
[GXF](https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html) is a framework from NVIDIA that provides a component-based architecture designed for developing hardware accelerated compute graphs. The framework is at the foundation of other high-performance SDKs such as [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), [DeepStream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/graphtools-docs/docs/text/GraphComposer_Graph_Runtime.html), and [Isaac ROS](https://github.com/NVIDIA-ISAAC-ROS). For example, [NITROS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros) (NVIDIA Isaac Transport for ROS) leverages GXF compute graphs embedded within ROS 2 nodes with optimized transport between them to achieve highly efficient ROS application graphs.

This repository contains the buildable source code for the GXF framework and its deployment for Isaac ROS NITROS. Scripts here will build the GXF source and install the updated binaries for all supported platforms in the [isaac_ros_gxf](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/tree/main/isaac_ros_gxf) package locally.

## Setup
The build environment can be setup on an x86_64 system running Ubuntu 20.04+ using the [Isaac ROS Dev)](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common) base containers.

## Build and Install
1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).

2. Clone the following repositories and its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    mkdir -p ~/workspaces/isaac_ros-dev/src && cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/gxf
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

3. Add a config file to update the environment image key with 'gxf' and add the directory to the Dockerfile search paths:

    ```bash
    echo "CONFIG_IMAGE_KEY=ros2_humble.user.gxf" >> ~/workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/.isaac_ros_common-config && \
    echo "CONFIG_DOCKER_SEARCH_DIRS=(../../gxf/docker)" >> ~/workspaces/isaac_ros-dev/src/isaac_ros_common/scripts/.isaac_ros_common-config
    ```
4. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

5. Inside the container, build and install GXF to `isaac_ros_gxf` package:

    ```bash
    cd /workspaces/isaac_ros-dev/src/gxf && \
      ./build_install_gxf_release.sh -i /workspaces/isaac_ros-dev/src/isaac_ros_nitros/isaac_ros_gxf
    ```
The scripts will build the source into a GXF release package and then install the headers and binaries into the 'isaac_ros_gxf' ROS 2 package used by NITROS at runtime.

## License
This work is published under the [NVIDIA ISAAC ROS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/blob/main/LICENSE) software license.

## Updates

| Date       | Changes                                                                                 |
| ---------- | --------------------------------------------------------------------------------------- |
| 2023-10-18 | Initial release                                                                         |
