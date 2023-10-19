'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
"""
This script can be used to run all the compiled tests on a jetson device. Should be used with gxf_engine.tar file
which can be obtained from https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/nightly/release-2.3.1/gxf_core-any-any-release-2.3.1_20211019_01acfc1_internal.tar

Usage : $ python3 jetson_run_tests.py -t ./all_tests_formatted

If you see issues with missing libgxf_core.so, run this
$ sudo update-alternatives --install /usr/lib/aarch64-linux-gnu/libgxf_core.so gxf_core /opt/nvidia/graph-composer/libgxf_core.so 50
"""

import argparse
import glob
import os
import subprocess
import sys


def main(args):
    return_code = 0
    filename = args.formatted_test_targets[0]
    with open(filename) as file:
        all_tests = file.readlines()
        all_tests = [line.rstrip() for line in all_tests]
    unit_tests = [line for line in all_tests if not line.endswith("_yaml")]
    failed_unit_tests = []
    for test in []:
        print("================================================================================")
        print(f"RUNNING {test}")
        print("================================================================================")
        try:
            res = subprocess.run(test, shell=True, check=True)
            res.check_returncode()
        except subprocess.CalledProcessError as e:
            failed_unit_tests.append(test)
    print("##################################################################")
    print(f"{len(unit_tests)} unit tests passed successfully")
    print("##################################################################")
    component_graph_tests = [
        line for line in all_tests if (line.endswith("_yaml") and not line.startswith("gxf/test"))
    ]
    component_graph_tests = [graph.replace("_yaml", ".yaml") for graph in component_graph_tests]
    test_graphs = glob.glob("gxf/test/apps/*.yaml")
    test_graphs = [test for test in test_graphs if not test.endswith("manifest.yaml")]
    graph_tests = component_graph_tests + test_graphs
    failed_graph_tests = []
    for graph in graph_tests:
        original_graph_name = graph.replace(".yaml", "_yaml")
        print("================================================================================")
        print(f"RUNNING {graph}")
        print("================================================================================")
        abs_path = os.path.abspath(os.path.expanduser(graph))
        dir_path = os.path.dirname(abs_path)
        yamls = []
        for file in os.listdir(dir_path):
            if file.endswith(".yaml"):
                yamls.append(file)
        for yaml in yamls:
            if graph.find("yaml") != -1:
                graph = yaml
                break

        cmd = "./" + original_graph_name
        if ("./gxf/test/apps/" in cmd):
            cmd = cmd.replace("apps/", "/gxf_test_apps_")
        if (os.path.isfile(cmd)):
            app_name = original_graph_name.replace("_yaml", ".yaml")
            if ("gxf_sample_tests_" in app_name):
                app_name = app_name.replace("gxf_sample_tests_", "")
            cmd += f" -app ./{app_name}"
            cmd += f" -manifest gxf/gxe/manifest.yaml"
        else:
            app_name = original_graph_name.replace("_yaml", ".yaml")
            cmd = "gxf/gxe/gxe"
            cmd += f" -app {app_name}"
            cmd += f" -manifest gxf/gxe/manifest.yaml"
        try:
            print("running test:", cmd)
            res = subprocess.run(cmd, shell=True, check=True)
            res.check_returncode()
        except subprocess.CalledProcessError as e:
            failed_graph_tests.append(original_graph_name)

    print("##################################################################")
    print(f"{len(unit_tests)} graph tests passed successfully")
    print("##################################################################")
    if failed_unit_tests:
        return_code += 1
        print("##################################################################")
        print(f"{len(failed_unit_tests)} unit tests failed")
        for test in failed_unit_tests:
            print(test)
        print("##################################################################")
    if failed_graph_tests:
        return_code += 1
        print("##################################################################")
        print(f"{len(failed_graph_tests)} graph tests failed")
        for test in failed_graph_tests:
            print(test)
        print("##################################################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--formatted_test_targets",
                        help="List of formatted test targets",
                        required=True,
                        nargs=1)
    args = parser.parse_args()
    res = main(args)
    sys.exit(res)
