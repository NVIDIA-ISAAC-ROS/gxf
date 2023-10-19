################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

""" Registry Dependency Manager
"""

import argparse
import os
import logging
import sys
import yaml
from yaml_loader import YamlLoader

def get_formatted_dep(dep):
    yaml_loader = YamlLoader()
    dep_metadata  = yaml_loader.load_yaml(dep)

    return {"extension": dep_metadata["name"],
            "uuid": dep_metadata["uuid"],
            "version": dep_metadata["version"],
            "manifest": dep}

def clean_path(path):
    return os.path.abspath(os.path.expanduser(path))

def clean_file_list(f_list):
    res = []
    for f in f_list:
        r = clean_path(f)
        if not os.path.exists(r):
            print(f"File not found {r}")
            sys.exit(1)
        res.append(r)

    return res

def update_dependencies(args):
    input_manifest = YamlLoader().load_yaml(args.input_manifest)
    if not input_manifest:
        return False

    output_deps = []
    # Copy the ngc dependencies and append local dependencies
    if input_manifest["dependencies"]:
        output_deps = input_manifest["dependencies"]

    for dep in args.deps:
        output_deps.append(get_formatted_dep(dep))
    input_manifest["dependencies"] = output_deps

    # Update absolute path for the extension library
    input_manifest["extension_library"] = clean_path(input_manifest["extension_library"])

    # Update absolute path for all file lists
    input_manifest["binaries"] = clean_file_list(input_manifest["binaries"])
    input_manifest["headers"] = clean_file_list(input_manifest["headers"])
    # input_manifest["python_bindings"] = clean_file_list(input_manifest["python_bindings"])
    input_manifest["python_sources"] = clean_file_list(input_manifest["python_sources"])

    with open(args.output_manifest, "wt") as output_manifest:
        yaml.dump(input_manifest, output_manifest, default_flow_style=False, sort_keys=False)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--deps",
                        help="List of metadatafiles from dependent extensions",
                        required=True, nargs="*")
    parser.add_argument("-mi", "--input_manifest",
                        help="Manifest file input", required=True)
    parser.add_argument("-mo", "--output_manifest",
                        help="Manifest file output", required=True)

    args = parser.parse_args()

    update_dependencies(args)
