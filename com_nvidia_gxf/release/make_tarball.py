#! /usr/env/python
# Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import yaml
import distro

LICENSE = """
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

FILEGROUP = """
filegroup(
  name = "{name}_release",
  srcs = ["{name}"],
  visibility = ["//visibility:public"],
)
"""

EXPORTS_FILES = """
exports_files(["{name}"])
"""

class Platform:
    """Describes a target platform architecture"""
    def __init__(self, content):
        self.platform = content["platform"]
        self.define = content["define"]
        self.src_lib = content["src_lib"]
        self.dst_lib = content["dst_lib"]
        self.bazel_config = content["bazel_config"]
        self.cxx_config = content["cxx_config"] if "cxx_config" in content else None
        self.disable_cxx11_abi = content["disable_cxx11_abi"] if "disable_cxx11_abi" in content else None

class Target:
    """Data and methods related to a Bazel target output"""
    def __init__(self, content, gxf_root, yaml_content, platform_dict):
        lib_path = content.split("/")

        self.name = lib_path[-1]
        self.dst = content
        self._libs = lib_path[-1]
        self._gxf_root = gxf_root
        self._release_dir = yaml_content["tarball_release_folder"]
        self._workspace_name = yaml_content["workspace_name_release"]
        self._platform_dict = platform_dict

    def copy_all_libs(self, lib_path, lib_out):
        """Copy target output to release directory"""
        root_lib = os.path.join(self._gxf_root, lib_path)
        dst = ''.join(self.dst)
        dst_dir = os.path.join(self._release_dir, lib_out, '/'.join(dst.split("/")[:-1]))

        path_lib = os.path.join(root_lib, "gxf", self.dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        if not os.path.exists(path_lib):
            print(f"File not found: {path_lib}")
            return False
        shutil.copy2(path_lib, dst_dir)

        return True

    def get_libs(self):
        return self._libs

def write_to_file(path, content):
    file_ex = os.path.exists(path)
    with open(path, "a") as f:
        if not file_ex:
            f.write(LICENSE)
        f.write(content)

def copy_files_for_platform(target_list, platform):
    """Build targets for one platform with Bazel and copy binaries to release folder"""
    bazel_opt = platform.bazel_config
    cmd = f"bazel build ... --config={bazel_opt}"
    if platform.cxx_config:
        cmd += f" --config={platform.cxx_config}"
    if platform.disable_cxx11_abi:
        cmd += f" --config=disable_cxx11_abi"
    print("###########################################################")
    print(f"Building Bazel Config {bazel_opt}")
    print(f"Command: {cmd}")
    print("###########################################################")
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    if not proc.returncode == 0:
        return False
    for target in target_list:
        if not target.copy_all_libs(platform.src_lib, platform.dst_lib):
            return False
    return True


def bazel_clean():
    cmd = "bazel clean --async"
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    if not proc.returncode == 0:
        return False
    return True


def write_lib_files(release_dir, target_list, platform_dict):
    """Build library files and copy to release folder"""
    all_libs_str = {}

    for target in target_list:
        tmp_all_libs = target.get_libs()
        all_libs_str[target.dst] = f"{tmp_all_libs}"

    bazel_clean()

    for platform in platform_dict.values():
        if not copy_files_for_platform(target_list, platform):
            print(f"Could not copy files for {platform.platform}")
            return False
        for dst, libs_str in all_libs_str.items():
            dir_build = os.path.join(release_dir, platform.dst_lib, '/'.join(dst.split("/")[:-1]))
            if not os.path.exists(dir_build):
                os.makedirs(dir_build)
            build_file = os.path.join(dir_build, "BUILD")
            files_with_extension = '.'.join(dst.split('.')[1:])
            if files_with_extension:
                content_to_write = FILEGROUP.format(name=libs_str)
            else:
                content_to_write = EXPORTS_FILES.format(name=libs_str)
            write_to_file(build_file, content_to_write)
    return True


def copy_files(release_dir:str, tocopy_list:list[dict]) -> bool:
    """
    Copy files from source or build directory to the installation directory.

    :param release_dir: The relative path to the installation directory
    :param tocopy_list: List of dictionaries where each element has the keys:
        - src: The origin filepath in the source or build directory
        - dst: The output filepath relative to the installation directory
    :return: "False" if the source file is not found, else "True"
    """
    for elm in tocopy_list:
        src = elm["src"]
        dst = elm["dst"]
        dst_dir_path = os.path.join(release_dir, os.path.dirname(dst))
        if not os.path.isdir(dst_dir_path):
            os.makedirs(dst_dir_path)
        dst_dir_file = os.path.join(dst_dir_path, os.path.basename(dst))
        if not os.path.isfile(src):
            print(f"Missing {src}")
            return False
        shutil.copy2(src, dst_dir_file)
    return True

def generate_code_coverage_report(release_dir):
    gen_cmd = f"bazel coverage --instrument_test_targets --combined_report=lcov \
                --java_runtime_version=remotejdk_11 //..."
    try:
        print("Generating Code coverage :", gen_cmd)
        res = subprocess.run(gen_cmd, shell=True, check=True)
        res.check_returncode()
    except subprocess.CalledProcessError as e:
        print(f"Code coverage report generation failed: {e}")
        return False

    staging_path =  tempfile.mkdtemp(prefix="/tmp/gxf.")
    html_cmd = f"genhtml bazel-out/_coverage/_coverage_report.dat --o {staging_path}"
    try:
        print("Generating Code Coverage HTML Report:", html_cmd)
        res = subprocess.run(html_cmd, shell=True, check=True)
        res.check_returncode()
    except subprocess.CalledProcessError as e:
        print(f"Code coverage HTML report generation failed: {e}")
        return False

    # create tarball to be added in the release package
    dst_path = os.path.join(release_dir, "gxf_code_coverage.tar.gz")
    with tarfile.open(dst_path, "w:gz") as tar:
        tar.add(staging_path, arcname=os.path.basename(staging_path))

    # cleanup
    shutil.rmtree(staging_path)
    return True

def match_distro_platform(distro_name_ubuntu, platform_name):
    if bool(distro_name_ubuntu):
        if "rhel" not in platform_name:
            print("Valid: distro - {0}, platform - {1}".format(distro.name(), platform_name))
            return True
        else:
            print("Invalid: distro - {0}, platform - {1}".format(distro.name(), platform_name))
            return False
    else:
        if "rhel" in platform_name:
            print("Valid: distro - {0}, platform - {1}".format(distro.name(), platform_name))
            return True
        else:
            print("Invalid: distro - {0}, platform - {1}".format(distro.name(), platform_name))
            return False

def make_tarball_release(root_gxf, yaml_content, single_platform:str=''):
    """Build and package the release tarball"""
    release_dir = yaml_content["tarball_release_folder"]
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)

    pathlib.Path(release_dir).mkdir(parents=True, exist_ok=True)

    distro_name_ubuntu = True if (distro.name().find("Ubuntu") != -1) else False
    platform_dict = {}
    platform_found = False
    for name, content in yaml_content["platforms"].items():
        if single_platform:
            if single_platform == name and match_distro_platform(distro_name_ubuntu, name) == True:
                platform_dict[name] = Platform(content)
                platform_found = True
                break
        else:
            if match_distro_platform(distro_name_ubuntu, name) == True:
                platform_dict[name] = Platform(content)
                platform_found = True

    if platform_found == False:
        print("Incorrect platform specified, tarball will not be built!")
        return False

    target_list = []
    for content in yaml_content["targets"]:
        target_list.append(Target(content, root_gxf, yaml_content,
                                  platform_dict))
    if not write_lib_files(release_dir, target_list, platform_dict):
        print("Error while writing library files, tarball will not be built!")
        return False

    for platform in platform_dict.values():
        platform_release_dir = os.path.join(release_dir, platform.dst_lib)
        if not copy_files(platform_release_dir, yaml_content["files_to_copy_per_platform"]):
            return False

    if not copy_files(release_dir, yaml_content["files_to_copy_release"]):
        return False
    copy_as_is = [{"src": elm, "dst": elm} for elm in yaml_content["files_to_copy_test_as_is"]]
    if not copy_files(release_dir, copy_as_is):
        return False

    # Code coverage report is now enabled in Jenkins CI
    # if bool(os.getenv('ENABLE_CODE_COVERAGE') or yaml_content["enable_code_coverage"]) \
    #     and not generate_code_coverage_report(release_dir):
    #     return False

    # Create a dummy coverity function file that gets loaded from gxf.bzl
    pathlib.Path(release_dir+"/coverity/bazel/").mkdir(parents=True, exist_ok=True)
    with open(release_dir+"/coverity/bazel/coverity.bzl","w") as coverity_file:
        copyright_message = [
            "\"\"\"\n"
            "Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.\n\n"
            "NVIDIA CORPORATION and its licensors retain all intellectual property\n"
            "and proprietary rights in and to this software, related documentation\n"
            "and any modifications thereto. Any use, reproduction, disclosure or\n"
            "distribution of this software and related documentation without an express\n"
            "license agreement from NVIDIA CORPORATION is strictly prohibited.\n"
            "\"\"\"\n"
        ]
        coverity_file.writelines(copyright_message)
        coverity_func = [
            "\n\n"
            "def coverity(name, tags = []):\n"
            "    pass\n"
        ]
        coverity_file.writelines(coverity_func)
    coverity_file.close()

    # Create an empty BUILD file
    bazel_file = open(release_dir+"/coverity/bazel/BUILD","w")
    bazel_file.close()

    # Update extension path in manifest file
    for platform in platform_dict.values():
        filename = yaml_content["manifest"]
        manifest_filename = filename[0].replace("gxf", platform.dst_lib, 1)
        with open(release_dir + "/" + manifest_filename, "w") as target_manifest_file:
            with open(filename[0], "r") as cur_manifest_file:
                extension_paths = cur_manifest_file.readlines()
                for extension_path in extension_paths:
                    new_extension_path = extension_path.replace("gxf", platform.dst_lib, 1)
                    target_manifest_file.write(new_extension_path)

    tarball_name = yaml_content["tarball_release_name"]
    os.system(f"tar zvcf {tarball_name} {release_dir}")
    print(f"Done.\nTarball release available at: {tarball_name}")
    shutil.rmtree(release_dir)
    return True


def check_keys(yaml_content):
    """Validate that manifest contains required keys"""
    keys = ["tarball_release_name",
            "tarball_release_folder",
            "workspace_name_release",
            "platforms",
            "targets",
            "files_to_copy_per_platform",
            "files_to_copy_release",
            "enable_code_coverage",
            "manifest"]
    is_ok = True
    for key in keys:
        if key not in yaml_content:
            print(f"Missing {key} in yaml file")
            is_ok = False
    return is_ok

def main():
    parser = argparse.ArgumentParser(
        description='Builds and packages a GXF release tarball'
    )
    parser.add_argument('yaml_file',
                        metavar='tarball_config.yaml',
                        type=str,
                        help='Manifest file describing release platform and file targets')
    parser.add_argument('tarball_release_name',
                        type=str,
                        help='Output tarball name')
    parser.add_argument('tarball_release_folder',
                        type=str,
                        help='Output tarball path to create')
    parser.add_argument('--single_platform',
                        type=str,
                        default='',
                        required=False,
                        help='Target platform string matching a known, named Platform specification')
    args = parser.parse_args()

    if not os.path.isfile(args.yaml_file):
        parser.print_help()
        print("<tarball_config.yaml> must be a file")
        sys.exit(1)
    with open(args.yaml_file) as f:
        yaml_str = f.read()
    yaml_obj = yaml.safe_load(yaml_str)
    yaml_obj["tarball_release_name"] = args.tarball_release_name
    yaml_obj["tarball_release_folder"] = args.tarball_release_folder
    if not check_keys(yaml_obj):
        parser.print_help()
        sys.exit(2)

    dir_script = os.path.dirname(__file__)
    root_gxf = os.path.abspath(os.path.join(os.path.curdir, dir_script, ".."))
    res = make_tarball_release(root_gxf, yaml_obj, args.single_platform)
    if not res:
        print("Could not create tarball")
        parser.print_help()
        sys.exit(3)


if __name__ == '__main__':
    main()
