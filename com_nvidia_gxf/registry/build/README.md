Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

# Importing Extensions from NGC using GXF registry

Extensions uploaded to NGC can be imported into bazel workspace using the `gxf_import_ext` rule provided in `gxf/registry/build/gxf_extension.bzl`. A sample example of the rule invocation have been shown in `gxf/registry/test/ngc/import/gxf_extension_test.bzl`.

## Prerequsites -
1. [Optional] GXF resgistry can be installed through the graph composer package. If the registry executable is not found, the bazel rule will download an internal build to perform the registry opertions.

2. [Mandatory] NGC api key. It can be imported into bazel workspace by either
    a. Export it as an env var "NGC_API_KEY"
    b. Install the ngc cli tool from -  https://ngc.nvidia.com/setup/installers/cli and set the ngc configuration. The bazel rule can extract the API key from the config file.

3. [Mandatory] Access to the required NGC private registry from which the extensions are to be imported. Ping Prashant Gaikwad (pgaikwad) for access.

## Instructions -

1. The bazel rule has been implemented as a reposiotry rule so any invocation of the rule must be done in the WORKSPACE file. See the `gxf_test_data` macro in WORKSPACE and its corresponding implementation in `gxf/third_party/gxf.bzl`

### Sample Usage -

```
    gxf_import_ext(
        name = "std_aarch64",
        ext_name = "StandardExtension",
        repo_name = "gxf-dev-team",
        arch = "aarch64",
        distribution = "ubuntu_20.04",
        os = "linux",
        version = "2.1.0",
        cuda = "11.4",
        tensorrt = "",
        vpi = "",
        build_file = clean_dep("//registry/test/ngc/import:standard_extension.BUILD"),
    )

```

2. The extensions downloaded from NGC are read as a filegroup in bazel using it's own BUILD file.

### Sample usage -

```
    filegroup(
        name = "standard_extension",
        srcs = ["libgxf_std.so"],
        visibility = ["//visibility:public"],
    )

```

3. To be used in gxf applications, the extensions can be imported in bazel workspace as a c++ target using `cc_import` rule.

### Sample Usage -

```
    alias(
        name = "libgxf_std.so",
        actual = select({
            "@com_nvidia_gxf//engine/build:cpu_host": "@std_x86_64_cuda_12_1//:standard_extension",
            "@com_nvidia_gxf//engine/build:cpu_aarch64": "@std_aarch64//:standard_extension",
        }),
        visibility = ["//visibility:public"],
    )


    cc_import(
        name = "std",
        shared_library = ":libgxf_std.so",
        tags = ["manual"],
        visibility = ["//visibility:public"],
    )

```

4. Extensions imported from NGC can also be used to run applications within bazel using the `nv_gxf_app` rule.

### Sample Usage -

```
    nv_gxf_app(
        name = "test_ping",
        srcs = ["test_ping_composer.yaml"],
        extensions = [
            "//registry/test/ngc:std",
            "//registry/test/ngc:test",
        ],
    )
```


5. The full example is setup in `gxf/registry/test/ngc/import`. Uncomment the import statements in ``gxf/third_party/gxf.bzl` and run the sample using `bazel run //registry/test/ngc/import:test_ping`