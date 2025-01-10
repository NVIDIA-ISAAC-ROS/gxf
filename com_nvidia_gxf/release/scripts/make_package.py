#! /usr/bin/python3
# Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
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
import argparse

EXPECTED_YAML_KEYS_CONTENT = ["required_files", "files_no_exec", "file_map"]
EXPECTED_YAML_KEYS_DEFINE = ["name", "version", "arch_x86", "arch_aarch64"]
EXPECTED_YAML_KEYS_DEFINE_GC = ["name", "version", "arch_x86"]


def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode


def copy_file(src, dst):
    dst_dir_path = os.path.dirname(dst)

    if not os.path.isdir(dst_dir_path):
        os.makedirs(dst_dir_path)
    if not os.path.isfile(src):
        if not os.path.isdir(src):
            print(f"Missing {src}")
            return False
        else:
            shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def copy_all_files(yaml_content, gxf_root, output_dir):
    file_map = yaml_content["file_map"]
    for tuple in file_map:
        src = os.path.join(gxf_root, tuple["src"])
        dst_val = tuple["dst"]
        if isinstance(dst_val, str):
            dst = os.path.join(output_dir, dst_val)
            res = copy_file(src, dst)
            if not res:
                return False
        elif isinstance(dst_val, list):
            for dst in dst_val:
                dst = os.path.join(output_dir, dst)
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

    parser = argparse.ArgumentParser(description='Create Graph composer and runtime packages.')
    parser.add_argument('yaml_content', metavar='yaml_content', type=str,
                        help='Path to the YAML content file')
    parser.add_argument('yaml_define', metavar='yaml_define', type=str,
                        help='Path to the YAML define file')
    parser.add_argument('gxf_root', metavar='gxf_root', type=str,
                        help='Path to the GXF root folder')
    parser.add_argument('output_dir', metavar='output_dir', type=str,
                        help='Path to the output folder')
    parser.add_argument('--rhel', action='store_true',
                        help='Create GC runtime rpm package for RHEL')
    parser.add_argument('--graphcomposer', action='store_true',
                        help='Create debian package with only Graph Composer. Default - Create debian package with Graph Composer and runtime ')
    args = parser.parse_args()

    if not os.path.isfile(args.yaml_content):
        print("<yaml_content> must be a file")
        return None
    yaml_content_path = expand_path(args.yaml_content)

    if not os.path.isfile(args.yaml_define):
        print("<yaml_define> must be a file")
        return None
    yaml_define_path = expand_path(args.yaml_define)

    if not os.path.isdir(args.gxf_root):
        print("<gxf_root> must be a folder")
        return None
    gxf_root = expand_path(args.gxf_root)

    if not os.path.isdir(args.output_dir):
        print("<output_dir> must be a folder")
        return None
    output_dir = expand_path(args.output_dir)

    graphcomposer = False
    if (args.graphcomposer):
        graphcomposer = True

    rhel_package = ''
    if(args.rhel):
        rhel_package = "rhel"

    yaml_content = get_yaml_content(yaml_content_path,
                                    EXPECTED_YAML_KEYS_CONTENT)
    if (graphcomposer):
        yaml_define = get_yaml_content(yaml_define_path, EXPECTED_YAML_KEYS_DEFINE_GC)
    else:
        yaml_define = get_yaml_content(yaml_define_path, EXPECTED_YAML_KEYS_DEFINE)

    if not (yaml_content and yaml_define):
        return None

    return yaml_content, yaml_define, gxf_root, output_dir, rhel_package, graphcomposer

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

def compose_rpm_name(name, version, arch):
    return f"{name}-{version}_{arch}.rpm"

def make_deb_pkg(pkg_name, path):
    cmd = f"dpkg-deb -v -D --root-owner-group --build {path} {pkg_name}"
    if run_cmd(cmd) != 0:
        print(f"Could not make {pkg_name}")
        return False
    return True

def make_rpm_pkg(pkg_name, path, spec, version):
    os.mkdir(f"{path}/rhel9/SOURCES")
    cmd = f"cd {path}/../ && tar zcf {path}/rhel9/SOURCES/graph_composer-{version}_el9_x86_64.tar.gz -C x86 opt && cd -"
    if run_cmd(cmd) != 0:
        print(f"Could not create a tar package {pkg_name}")
        return False
    cmd = f"cd {path} && rpmbuild --define \"_topdir {path}/rhel9\" -ba {path}/rhel9/SPECS/{spec}"
    if run_cmd(cmd) != 0:
        print(f"Could not make {pkg_name}")
        return False
    return True

def main():
    script_dir = parse_args()
    if not script_dir:
        return 1
    yaml_content, yaml_define, gxf_root, output_dir, rhel_package, graphcomposer = script_dir

    if not copy_all_files(yaml_content, gxf_root, output_dir):
        return 2
    if not check_files_exist(yaml_content, output_dir):
        return 2
    if not set_files_mod(yaml_content, output_dir):
        return 2

    if rhel_package:
        path_x86 = os.path.join(output_dir, "x86")
        name_pkg_x86 = compose_rpm_name(yaml_define["name"],
                                        yaml_define["version"] + "_el9",
                                        yaml_define["arch_x86"])
        if not make_rpm_pkg(name_pkg_x86, path_x86, "graph_composer_runtime.spec", yaml_define["version"]):
            return 2
    elif not graphcomposer:
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
