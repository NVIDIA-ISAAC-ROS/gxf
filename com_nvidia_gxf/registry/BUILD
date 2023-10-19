# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

load("//registry/build/packaging:packaging.bzl", "registry_package")
"""
# DO NOT USE, PYARMOR MISSING, please use registry/build/packaging/make_bin_and_pip_pkg.py instead
registry_package(
    name = "registry",
    version = "1.1.0",
    description = "Registry package",
    file_description = "registry/README",
    classifiers = [],
    keywords = "registry",
    url = "http://www.nvidia.com",
    author = "Nvidia",
    author_email = "",
    license = "NVIDIA proprietary",
    path_main_file = "registry.cli.registry_cli:main",
    install_requires = ["packaging","pyyaml", "result", "requests", "toml"],
    deps = [
        "//registry:README.public",
        "//registry:__init__.py",
        "//registry/cli:__init__.py",
        "//registry/cli:cli_argparse_group_maker.py",
        "//registry/cli:cli_pretty_format.py",
        "//registry/cli:cli_query_maker.py",
        "//registry/cli:registry_cli.py",
        "//registry/core:__init__.py",
        "//registry/core:component.py",
        "//registry/core:config.py",
        "//registry/core:core.py",
        "//registry/core:core_interface.py",
        "//registry/core:dependency_governer.py",
        "//registry/core:dependency_manager.py",
        "//registry/core:extension.py",
        "//registry/core:ngc_client.py",
        "//registry/core:packager.py",
        "//registry/core:parameter.py",
        "//registry/core:repository.py",
        "//registry/core:ngc_repository.py",
        "//registry/core:repository_manager.py",
        "//registry/core:database.py",
        "//registry/core:utils.py",
        "//registry/core:yaml_loader.py",
        "//registry/core:logger.py",
        "//registry/bindings:pygxf.so"
    ],

    packages = [
        "registry/cli",
        "registry/core",
        "registry/bindings",
     ],
     data = ["registry/bindings/pygxf.so"]
)
"""
