#! /usr/bin/python3
# Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import shutil
import subprocess
import sys
from multiprocessing import Process
import io
import yaml

BUILD_REGISTRY_BIN = "BUILD_REGISTRY_BIN"
BUILD_REGISTRY_PIP = "BUILD_REGISTRY_PIP"

PYARMOR_INCUDE = "from registry.pytransform import pyarmor_runtime"

RUN_PY = """
import sys
{pyarmor_include}
from registry.cli import registry_cli

def main():
    registry_cli.main(sys.argv[1:])

if __name__ == "__main__":
    main()
"""

_SETUP_PY_TEMPLATE_PYARMOR = """
from setuptools import setup

setup(
    name = "{reg}",
    version = "{ver}",
    description = "Command line interface to interact with registry",
    long_description = "",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    keywords = ["registry"],
    url = "https://www.nvidia.com",
    author = "NVIDIA",
    author_email = "",
    license = "NVIDIA proprietary",
    package_data ={{"registry/bindings": ["pygxf.so"],
    "registry/pytransform": ["_pytransform.so"]}},
    install_requires = ["packaging==23.1","pyyaml==5.3.1", "result==0.10.0", "requests==2.25.1", "toml==0.10.2"],
    packages=["registry/cli", "registry/core", "registry/bindings",
              "registry/pytransform", "registry"],
    entry_points = {{"console_scripts": "{reg} = registry.run:main"}},
    platforms=["{platform}"],
    zip_safe=False,
)
"""

_SETUP_PY_TEMPLATE_NO_PYARMOR = """
from setuptools import setup

setup(
    name = "{reg}",
    version = "{ver}",
    description = "Command line interface to interact with registry",
    long_description = "",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    keywords = ["registry"],
    url = "https://www.nvidia.com",
    author = "NVIDIA",
    author_email = "",
    license = "NVIDIA proprietary",
    package_data ={{"registry/bindings": ["pygxf.so"]}},
    install_requires = ["packaging==23.1","pyyaml==5.3.1", "result==0.10.0", "requests==2.25.1", "toml==0.10.2"],
    packages=["registry/cli", "registry/core", "registry/bindings", "registry"],
    entry_points = {{"console_scripts": "{reg} = registry.run:main"}},
    platforms=["{platform}"],
    zip_safe=False,
)
"""

hidden_imports = ["registry.cli.cli_argparse_group_maker",
                  "registry.cli.registry_cli",
                  "registry.cli.cli_pretty_format",
                  "registry.cli.cli_query_maker",
                  "registry.core.core_interface",
                  "registry.core.ngc_client",
                  "registry.core.packager",
                  "registry.core.component",
                  "registry.core.dependency_manager",
                  "registry.core.parameter",
                  "registry.core.config",
                  "registry.core.yaml_loader",
                  "registry.core.core",
                  "registry.core.extension",
                  "registry.core.logger",
                  "registry.core.database",
                  "registry.core.repository",
                  "registry.core.ngc_repository",
                  "registry.core.repository_manager",
                  "registry.core.utils",
                  "registry.core.dependency_governer",
                  "registry.core.version",
                  "registry.bindings",
                  "registry.bindings.pygxf",
                  "logging",
                  "logging.handlers",
                  "pathlib", "typing", "result",
                  "yaml", "toml", "requests", "sqlite3", "packaging",
                  ]


def parse_input():
    args = sys.argv[1:]
    if len(args) != 3:
        return 1
    gxf_root = args[0]
    if not os.path.isdir(gxf_root):
        print("<gxf_root> must be a folder")
        return 1
    pygxf_so = args[1]
    if not os.path.isfile(pygxf_so):
        print("<pygxf.so> must be a file")
        return 1
    yaml_file = args[2]
    if not os.path.isfile(yaml_file):
        print("<build_setting.yaml> must be a file")
        return 1
    with open(yaml_file, "r") as f:
        content = f.read()
        yaml_content = yaml.safe_load(content)
    expected_keys = ["pip_platform", "enable_pyarmor", "registry",
                     "registry_version"]
    for key in expected_keys:
        if key not in yaml_content:
            print(f"Missing key: {key} in {yaml_file}")
            return 1

    return gxf_root, pygxf_so, yaml_content


def prepare_files(gxf_root, pygxf_so, yaml_content):
    enable_pyarmor = yaml_content["enable_pyarmor"]
    f = open(os.path.join(os.path.dirname(sys.argv[0]), "allowed_files.txt"))
    trim_nl = lambda w: w[:-1] if w[-1] == "\n" else w
    allowed_files = [trim_nl(elm) for elm in f.readlines()]
    f.close()
    if os.path.exists(BUILD_REGISTRY_BIN):
        shutil.rmtree(BUILD_REGISTRY_BIN)
    os.makedirs(BUILD_REGISTRY_BIN)
    if os.path.exists(BUILD_REGISTRY_PIP):
        shutil.rmtree(BUILD_REGISTRY_PIP)
    os.makedirs(BUILD_REGISTRY_PIP)
    for file in allowed_files:
        directory = os.path.join(BUILD_REGISTRY_BIN, os.path.dirname(file))
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_loc = os.path.join(gxf_root, file)
        shutil.copy2(file_loc, directory)

    dir_bindings = os.path.join(BUILD_REGISTRY_BIN, "registry/bindings")
    os.makedirs(dir_bindings)
    shutil.copy2(pygxf_so, dir_bindings)
    process = subprocess.Popen(f"touch {dir_bindings}/__init__.py".split())
    process.wait()

    os.chdir(BUILD_REGISTRY_BIN)
    if enable_pyarmor:
        process = subprocess.Popen(
            "pyarmor obfuscate --recursive --output dist/registry registry/__init__.py",
            shell=True)
        process.wait()
        process = subprocess.Popen("cp -r dist/registry/* registry/",
                                   shell=True)
        process.wait()
        shutil.rmtree("dist")

    pyarmor_include = PYARMOR_INCUDE if enable_pyarmor else ""
    f = open("registry/run.py", "w")
    f.write(RUN_PY.format(pyarmor_include=pyarmor_include))
    f.close()
    os.chdir("..")
    process = subprocess.Popen(f"cp -r {BUILD_REGISTRY_BIN}/* "
                               f"{BUILD_REGISTRY_PIP}", shell=True)
    process.wait()


def make_pip_package(yaml_content):
    pip_platform = yaml_content["pip_platform"]
    enable_pyarmor = yaml_content["enable_pyarmor"]
    registry = yaml_content["registry"]
    registry_version = yaml_content["registry_version"]
    os.chdir(BUILD_REGISTRY_PIP)
    with open("setup.py", "w") as f:
        setup_py_tplt = _SETUP_PY_TEMPLATE_PYARMOR if enable_pyarmor \
            else _SETUP_PY_TEMPLATE_NO_PYARMOR
        f.write(setup_py_tplt.format(reg=registry,
                                     ver=registry_version,
                                     platform=pip_platform))

    package_name = f"{registry}-{registry_version}-py3-none-{pip_platform}.whl"
    cmd = f"python3 setup.py bdist_wheel --plat-name {pip_platform}"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        exit(1)
    cmd = f"cp dist/{package_name} ../{package_name}"
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        exit(1)
    os.chdir("..")
    shutil.rmtree(BUILD_REGISTRY_PIP)


def make_standalone_bin(yaml_content):
    enable_pyarmor = yaml_content["enable_pyarmor"]
    registry = yaml_content["registry"]
    os.chdir(BUILD_REGISTRY_BIN)
    data = ["registry/pytransform/_pytransform.so"] if enable_pyarmor else []
    data_dirs = [os.path.dirname(elm) for elm in data]
    cmd = "pyinstaller --onefile registry/run.py --hidden-import pkg_resources.extern"
    for i in range(len(data)):
        cmd += f" --add-data \"{data[i]}:{data_dirs[i]}\""

    for elm in hidden_imports:
        cmd += f" --hidden-import {elm} "
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        exit(1)
    process = subprocess.Popen(f"cp dist/run ../{registry}", shell=True)
    process.wait()
    if process.returncode != 0:
        exit(1)
    os.chdir("..")
    shutil.rmtree(BUILD_REGISTRY_BIN)


def main():
    res = parse_input()
    if isinstance(res, int):
        print(f"Usage:\n{sys.argv[0]} <gxf_root> <pygxf.so> <build_setting.yaml>")
        sys.exit(res)

    gxf_root, pygxf_so, yaml_content = res
    prepare_files(gxf_root, pygxf_so, yaml_content)
    p1 = Process(target=make_standalone_bin, args=(yaml_content,))
    p2 = Process(target=make_pip_package, args=(yaml_content,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    if p1.exitcode == 0:
        print("Successfully built standalone binary")
    else:
        print("Error while building standalone binary")
    if p2.exitcode == 0:
        print("Successfully built pip package")
    else:
        print("Error while building pip package")
    return (p1.exitcode + p2.exitcode)


if __name__ == "__main__":
    res = main()
    sys.exit(res)
