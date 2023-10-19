#! /usr/bin/python3
# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import os
import subprocess
import sys

SOURCE_NAME = "gxf_cli.py"
BINARY_NAME = "gxf_cli"

def make_single_binary(root_path):
    source = os.path.join(root_path, SOURCE_NAME)
    cmd = f"python3 -m PyInstaller --onefile -p {root_path} --clean --name {BINARY_NAME} {source}"
    process  = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        raise Exception("GXF CLI Binary Generation Failure")

if __name__ == "__main__":
    gxf_cli_root = os.path.abspath(sys.argv[1])
    try:
        make_single_binary(gxf_cli_root)
    except Exception as e:
        print(e)
        exit(1)