
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA Corporation and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA Corporation is strictly prohibited.

# Pre-requisites:
Install dependencies
$./engine/build/scripts/install_dependencies.sh

# Build Instructions:
$bazel build //registry/...

# Usage
Some commands here to help you publish extensions, It uses the the registry target in bazel instead of the registry binary.

Clean registry cache
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- cache -c
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- repo clean
```
 Build and register all extensions and variants
```


bazel build --config=x86_64_cuda_12_2

```

List extensions
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn list
```

List extension variants
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn variants --help
```

To add a new ngc repo from nv-gxf-dev org and gx-dev team to local config file
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- repo add ngc -n gxf-dev -o nv-gxf-dev -t gxf-dev -a <apikey>
```

Extension publish

Each extension has an interface and multiple variants. Interfaces are published first and then all the variants.

```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish interface --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish interface -n <extn-name> -r <repo-name == gxf-dev>
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn variants --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish variant --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish variant -n <name> -r <repo-name> -a x86_64 -f ubuntu_18.04 -o linux -c cuda-10.2
```

Interface and all variants of an extension can be published at once using
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish one --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish one -n <name> -r <repo-name>
```

If you are planning to build and publish multiple extensions and variants you can publish them all at once using
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish all --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn publish all -r <repo-name>
```

Extension remove

An extension and its variants can be removed from an NGC. To remove the extension, sync the corresponding NGC repository.
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- repo sync -n <repo-name>
```

Each extension has an interface and multiple variants. All variants need to be removed first before remove any interface.

```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove interface --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove interface -n <extn-name> -r <repo-name == gxf-dev>
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn variants --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove variant --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove variant -n <name> -r <repo-name> -a x86_64 -f ubuntu_18.04 -o linux -c cuda-10.2
```

Interface and all variants of an extension can be removed at once using
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove one --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove one -n <name> -s <version>
```

All extensions and variants can be removed from a team at once using
```
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove all --help
bazel run @com_nvidia_gxf//registry/cli:registry_cli -- extn remove all -r <repo-name>
```
