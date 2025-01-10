################################################################################
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

# Creating a GXF Binary Release

## `make_tarball.py`

The `make_tarball.py` script builds and packages a release tarball with
GXF binaries, headers, and external build rules. Use the `--help` option
for a full list of arguments:

```sh
python3 release/make_tarball.py --help
usage: make_tarball.py [-h] [--single_platform SINGLE_PLATFORM] tarball_config.yaml tarball_release_name tarball_release_folder

Builds and packages a GXF release tarball

positional arguments:
  tarball_config.yaml   Manifest file describing release platform and file targets
  tarball_release_name  Output tarball name
  tarball_release_folder
                        Output tarball path to create

options:
  -h, --help            show this help message and exit
  --single_platform SINGLE_PLATFORM
                        Target platform string matching a known, named Platform specification
```

## Packaging Instructions

Example command to build and use a release package:

1) Build using a command similar to below. `tmp/packaging` can be replaced for a more convenient location, if needed.
```sh
$ python3 release/make_tarball.py release/tarball_content.yaml release_tarball /tmp/packaging/
```

2) Extract the release_tarball generated in the previous step
```
$ tar xvf release_tarball --directory /
```
The step above places the release package in `/tmp/packaging` directory.

# Using a GXF Binary Release

1) Make sure that the dependencies are installed/setup correctly. The following are sample paths for gxf_core and gxe.

```
$ update-alternatives --query gxf_core

Name: gxf_core
Link: /usr/lib/x86_64-linux-gnu/libgxf_core.so
Status: auto
Best: /opt/nvidia/graph-composer/libgxf_core.so
Value: /opt/nvidia/graph-composer/libgxf_core.so

Alternative: /opt/nvidia/graph-composer/libgxf_core.so
Priority: 50

$ update-alternatives --query gxe
Name: gxe
Link: /usr/bin/gxe
Status: auto
Best: /opt/nvidia/graph-composer/gxe
Value: /opt/nvidia/graph-composer/gxe

Alternative: /opt/nvidia/graph-composer/gxe
Priority: 50
```

4) Run the `<release_build_installation_path/release>/release_build_setup.sh` script

5) The release package (installed in `/tmp/packaging` in the case described) is ready to be used now.
