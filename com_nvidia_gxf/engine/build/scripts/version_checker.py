'''
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import sys
import argparse
import subprocess

packages = {
    "OS": {
        "host": "Ubuntu 20.04.4 LTS",
        "jetson": "Ubuntu 20.04.4 LTS",
        "cmd": "lsb_release -d | cut -f 2"
    },
    "Bazel": {
        "host": "2.2.0",
        "jetson": "N/A",
        "cmd": "bazel version | grep 'Build label' | awk '{print $3}'"
    },
    "GPU_Driver": {
        "host": ">=418",
        "jetson": "N/A",
        "cmd": "cat /proc/driver/nvidia/version | grep NVRM |awk '{print $8}'"
    },
    "Cuda": {
        "host": "10.x.x",
        "jetson": "10.x.x",
        "cmd": "/usr/local/cuda/bin/nvcc --version | grep release | awk '{print $NF}' | cut -c 2-"
    },
    "Cudnn": {
        "host": "7.6.x.x",
        "jetson": "7.6.x.x",
        "cmd": "dpkg --list | grep 'libcudnn7 ' | awk '{print $3}' | cut -d '-' -f1"
    },
    "pycapnp": {
        "host": ">=0.6.3",
        "jetson": ">=0.6.3",
        "cmd": "pip3 list | grep pycapnp | awk '{print $2}'"
    },
    "librosa": {
        "host": ">=0.6.3",
        "jetson": "N/A",
        "cmd": "pip3 list | grep librosa | awk '{print $2}'"
    },
    "SoundFile": {
        "host": ">=0.10.2",
        "jetson": "N/A",
        "cmd": "pip3 list | grep SoundFile | awk '{print $2}'"
    },
    "Python2": {
        "host": "2.7.x",
        "jetson": "2.7.x",
        "cmd": "python2 -V 2>&1 | awk '{print $2}'"
    },
    "Python3": {
        "host": "3.6.x",
        "jetson": "3.6.x",
        "cmd": "python3 -V 2>&1 | awk '{print $2}'"
    }
}

# Function to show comparative results on console
def show():
    print("-------"*9)
    print("|%-20s|%-20s|%-20s|" % ("Package", "Recommended Version", "Current Version"))
    print("-------"*9)
    for package in packages:
        if not packages[package][device] == "N/A":
            print("|%-20s|%-20s|%-20s|" % (package,
                                           packages[package][device], packages[package]["version"]))
    print("-------"*9)

# Function to run command based on device type(Host or Jetson) and store the output
def run():
    for package in packages:
        if device == "jetson":
            cmd = "sshpass -p " + jetson_arg.password + " ssh " + jetson_arg.username + \
                "@" + jetson_arg.ip + " " + packages[package]["cmd"]
        else:
            cmd = packages[package]["cmd"]      # For host

        p = subprocess.Popen(
            [cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        version, _ = p.communicate()
        if not version == b"":
            version = version.decode('utf-8').strip()
        else:
            version = "N/A"
        packages[package]["version"] = version


# Help: python3 engine/build/scripts/version_checker.py -h
# For Host: python3 engine/build/scripts/version_checker.py
# For Jetson: python3 engine/build/scripts/version_checker.py -i <jetson_ip> -u <jetson_username> -p <jetson_password>
if __name__ == "__main__":
    device = "host"

    if (len(sys.argv) > 1):
        device = "jetson"
        parser = argparse.ArgumentParser(
            description="Check package version dependencies for Host & Jetson")
        parser.add_argument('-i', '--ip', type=str,
                            action="store", dest="ip", help="Jetson IP")
        parser.add_argument('-u', '--user', type=str, action="store",
                            dest="username", help="Jetson Username")
        parser.add_argument('-p', '--password', type=str,
                            action="store", dest="password", help="Jetson Password")
        jetson_arg = parser.parse_args()

    run()
    show()
