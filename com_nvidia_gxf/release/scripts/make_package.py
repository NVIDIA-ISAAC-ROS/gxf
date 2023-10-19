#! /usr/bin/python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import subprocess
import sys
import io
import yaml
import shutil

EXPECTED_YAML_KEYS_CONTENT = ["required_files", "files_no_exec", "file_map"]
EXPECTED_YAML_KEYS_DEFINE = ["name", "version", "arch_x86", "arch_aarch64"]


def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode


def copy_file(src, dst):
    dst_dir_path = os.path.dirname(dst)
    if not os.path.isdir(dst_dir_path):
        os.makedirs(dst_dir_path)
    if not os.path.isfile(src):
        print(f"Missing {src}")
        return False
    shutil.copy2(src, dst)
    return True


def copy_all_files(yaml_content, gxf_root, output_dir):
    file_map = yaml_content["file_map"]
    for tuple in file_map:
        src = os.path.join(gxf_root, tuple["src"])
        dst = os.path.join(output_dir, tuple["dst"])
        res = copy_file(src, dst)
        if not res:
            return False
    return True


def get_yaml_content(yaml_path, expected_keys):
    with open(yaml_path, "r") as f:
        f_content = f.read()
        yaml_content = yaml.safe_load(f_content)
    for elm in expected_keys:
        if elm not in yaml_content:
            print(f"Missing key: {elm} in {yaml}")
            return None
    return yaml_content


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def parse_args():
    args = sys.argv[1:]

    if len(args) != 4:
        print("Invalid argument number")
        return None

    if not os.path.isfile(args[0]):
        print("<yaml_content> must be a file")
        return None
    yaml_content_path = expand_path(args[0])

    if not os.path.isfile(args[1]):
        print("<yaml_define> must be a file")
        return None
    yaml_define_path = expand_path(args[1])

    if not os.path.isdir(args[2]):
        print("<gxf_root> must be a folder")
        return None
    gxf_root = expand_path(args[2])

    if not os.path.isdir(args[3]):
        print("<output_dir> must be a folder")
        return None
    output_dir = expand_path(args[3])

    yaml_content = get_yaml_content(yaml_content_path,
                                    EXPECTED_YAML_KEYS_CONTENT)
    yaml_define = get_yaml_content(yaml_define_path, EXPECTED_YAML_KEYS_DEFINE)
    if not (yaml_content and yaml_define):
        return None

    return yaml_content, yaml_define, gxf_root, output_dir


def check_files_exist(yaml_content, gxf_root):
    for file in yaml_content["required_files"]:
        fp = os.path.join(gxf_root, file)
        if not os.path.isfile(fp):
            print(f"Missing {fp}")
            return False
    return True


def set_files_mod(yaml_content, output_dir):
    command_755 = f"chmod -R 755 {output_dir}"
    if run_cmd(command_755):
        print("Could not set file mod 755")
        return False
    files_no_exec = [os.path.join(output_dir, elm)
                     for elm in yaml_content["files_no_exec"]]
    command_644 = f"chmod 644 " + " ".join(files_no_exec)
    if run_cmd(command_644):
        print("Could not set file mod to 644")
        return False
    return True


def compose_deb_name(name, version, arch):
    return f"{name}-{version}_{arch}.deb"


def make_deb_pkg(pkg_name, path):
    cmd = f"dpkg-deb -v -D --root-owner-group --build {path} {pkg_name}"
    if run_cmd(cmd) != 0:
        print(f"Could not make {pkg_name}")
        return False
    return True


def main():
    script_dir = parse_args()
    if not script_dir:
        return 1
    yaml_content, yaml_define, gxf_root, output_dir = script_dir

    if not copy_all_files(yaml_content, gxf_root, output_dir):
        return 2
    if not check_files_exist(yaml_content, output_dir):
        return 2
    if not set_files_mod(yaml_content, output_dir):
        return 2

    name_pkg_aarch64 = compose_deb_name(yaml_define["name"],
                                        yaml_define["version"],
                                        yaml_define["arch_aarch64"])
    path_jetson = os.path.join(output_dir, "jetson")
    if not make_deb_pkg(name_pkg_aarch64, path_jetson):
        return 2

    dst_jetson_deb = os.path.join(output_dir, "x86/opt/nvidia/graph-composer")
    copy_file(name_pkg_aarch64, dst_jetson_deb)

    path_x86 = os.path.join(output_dir, "x86")
    name_pkg_x86 = compose_deb_name(yaml_define["name"],
                                    yaml_define["version"],
                                    yaml_define["arch_x86"])
    if not make_deb_pkg(name_pkg_x86, path_x86):
        return 2

    print("Done.")
    return 0


if __name__ == "__main__":
    res = main()
    if res == 1:
        print(
            f"Usage:\n{sys.argv[0]} <yaml_content> <yaml_define> <gxf_root> <output_dir>")
        exit(res)
    exit(res)
