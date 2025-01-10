# /usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import os
import subprocess
import uuid

def get_uid():
   uid = subprocess.check_output("printf '0x%s\n' $(uuidgen -r | tr -d '-' | cut -b 1-16)",
                                 shell=True)
   return str(uid).rstrip("\n")

def write_bazelrc(args):
  text = """
build --nokeep_going --color=yes -c opt --crosstool_top=@toolchain//crosstool:toolchain

build --define=target_platform=x86_64
build --action_env=target_platform="x86_64"
build:x86_64 --define=target_platform=x86_64
build:x86_64 --action_env=target_platform="x86_64"

build:x86_64_rhel9 --define=target_platform=x86_64_rhel9
build:x86_64_rhel9 --action_env=target_platform="x86_64_rhel9"

build:jetson --cpu=aarch64
build:jetson --define=target_platform=jetson
build:jetson --action_env=target_platform="jetson"

build:debug -c dbg --strip="never"
  """
  with open(os.path.abspath("{0}/{1}/.bazelrc".format(args.output_dir, args.extn_name)),'w') as file:
    file.write(text)

def write_workspace(args):
  text = """
_workspace_name = "{0}"
workspace(name = _workspace_name)

local_repository(
    name = "com_extension_dev",
    path = "/opt/nvidia/graph-composer/extension-dev"
)

load(
    "@com_extension_dev//build:graph_extension.bzl",
    "graph_nvidia_extension",
)

load("@com_extension_dev//build/toolchain:toolchain.bzl", "toolchain_configure")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

toolchain_configure(name = "toolchain")

# Package created from - https://github.com/jbeder/yaml-cpp.git
# commit = "9a3624205e8774953ef18f57067b3426c1c5ada6"
http_archive(
    name = "yaml-cpp",
    build_file = "@com_extension_dev//build:third_party/yaml-cpp.BUILD",
    strip_prefix = "yaml-cpp-9a3624205e8774953ef18f57067b3426c1c5ada6",
    url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/yaml-cpp/9a3624205e8774953ef18f57067b3426c1c5ada6.tar.gz",
    sha256 = "e39f54bd2927692603378e373009e56b4891701cee8af7c27370c36978a43ffa",
    type = "tar.gz",
)

# Package created from - https://github.com/bazelbuild/rules_python
# version - 0.1.0
http_archive(
    name = "rules_python",
    url = "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/bazelbuild/rules_python/rules_python-0.1.0.tar.gz",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
)

# Package created from - https://github.com/Neargye/magic_enum
# version - 0.9.3
http_archive(
    name = "magic_enum",
    strip_prefix = "magic_enum-0.9.3",
    urls = ["https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/dependencies/external/magic_enum/v0.9.3.zip"],
)

graph_nvidia_extension(
    name = "StandardExtension",
    version = "2.6.0",
)
  """.format(args.extn_name)
  with open(os.path.abspath("{0}/{1}/WORKSPACE".format(args.output_dir, args.extn_name)),'w') as file:
    file.write(text)

def write_app(args, extnUuid):
  text = """
%YAML 1.2
---
dependencies:
- extension: {1}
  uuid: {2}
  version: 1.0.0
---
name: source
components:
- name: signal
  type: sample::{1}::{0}
- type: nvidia::gxf::CountSchedulingTerm
  parameters:
    count: 10
---
components:
- type: nvidia::gxf::GreedyScheduler
  parameters:
    realtime: false
    max_duration_ms: 1000000
  """.format(args.codelet_name, args.extn_name, extnUuid)
  with open("{}/app.yaml".format(os.path.abspath("{0}/{1}/apps".format(args.output_dir, args.extn_name))),'w') as file:
    file.write(text)


def write_source(args, extnUuid):
  extn_name = args.extn_name
  codelet_name = args.codelet_name
  filename = os.path.abspath("{0}/{1}/extensions/{1}".format(args.output_dir, args.extn_name))
  print("write source ", filename)
  text = """
#pragma once

#include "gxf/std/codelet.hpp"

namespace sample {{
namespace {1} {{

// Logs a message in start() and tick()
class {0} : public nvidia::gxf::Codelet {{
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override {{ return GXF_SUCCESS; }}
}};

}}  // namespace {1}
}}  // namespace sample
  """.format(codelet_name, extn_name)
  with open("{0}/{1}.hpp".format(filename, args.codelet_name),'w') as file:
    file.write(text)

  text = """
#include "{0}.hpp"  // NOLINT

namespace sample {{
namespace {1} {{

gxf_result_t {0}::start() {{

    GXF_LOG_INFO("{0}::start");
    return GXF_SUCCESS;
}}

gxf_result_t {0}::tick() {{
  GXF_LOG_INFO("{0}::tick");
  return GXF_SUCCESS;
}}

}}  // namespace {1}
}}  // namespace sample

  """.format(codelet_name, extn_name)

  codeletUuid = str(uuid.uuid3(uuid.uuid4(), args.extn_name))

  with open("{0}/{1}.cpp".format(filename, args.codelet_name),'w') as file:
    file.write(text)
  text = """
#include "{0}.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x{2}, 0x{3}, "{1}",
                         "A Dummy Example", "", "1.0.0", "NVIDIA");
GXF_EXT_FACTORY_ADD(0x{4}, 0x{5}, sample::{1}::{0},
                    nvidia::gxf::Codelet, "Dummy example source codelet.");
GXF_EXT_FACTORY_END()
  """.format(codelet_name, extn_name, extnUuid.replace("-","")[:16],
          extnUuid.replace("-","")[16:], codeletUuid.replace("-","")[:16],
          codeletUuid.replace("-","")[16:])
  with open("{0}/{1}.cpp".format(filename, args.extn_name),'w') as file:
    file.write(text)

  text = """
load("@com_extension_dev//build:graph_extension.bzl", "graph_cc_extension")
load("@com_extension_dev//build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

graph_cc_extension(
    name = "{0}",
    srcs = [
        "{1}.cpp",
        "{0}.cpp",
    ],
    hdrs = [
        "{1}.hpp",
    ],
    deps = [
        "@StandardExtension",
    ],
)

register_extension(
    name = "register_{0}_ext",
    badges = [""],
    extension = "{0}",
    labels = [
        "nvidia",
        "gpu",
    ],
    ngc_dependencies = {{
        "StandardExtension": "2.6.0",
    }},
    priority = "1",
    git_repository = "",
    url = "www.example.com",
    uuid = "{2}",
    version = "1.0.0",
    visibility = ["//visibility:public"],
)
""".format(extn_name, codelet_name, str(extnUuid))
  with open("{0}/BUILD".format(filename),'w') as file:
    file.write(text)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser("Graph Extension Generator")
  parser.add_argument("--output_dir", type=str, help="output directory")
  parser.add_argument("--extn_name", type=str, help="extension name")
  parser.add_argument("--codelet_name", type=str, help="codelet name")

  args = parser.parse_args()

  if not args.extn_name or not args.codelet_name:
    parser.print_help()
    sys.exit(-1)

  extnUuid = str(uuid.uuid3(uuid.uuid4(), args.extn_name))

  extnPath = os.path.join(args.output_dir, args.extn_name)
  codeletPath = os.path.join(extnPath, "extensions", args.extn_name)

  try:
      os.makedirs(extnPath, mode=0o777, exist_ok=True)
      os.makedirs(os.path.join(extnPath,"apps"), mode=0o777, exist_ok=True)
      os.makedirs(os.path.join(codeletPath), mode=0o777, exist_ok=True)
  except OSError as e:
      print(f"Failed to create directory")
      print(f"Error: {e}")
      sys.exit(-1)

  os.system("touch {0}/WORKSPACE {0}/.bazelrc".format(extnPath))
  os.system("touch {0}/apps/app.yaml".format(extnPath))
  os.system("touch {0}/BUILD {0}/{1}.cpp {0}/{2}.cpp \
             {0}/{2}.hpp".format(codeletPath, args.extn_name, args.codelet_name))

  write_workspace(args)
  write_bazelrc(args)
  write_app(args, extnUuid)
  write_source(args, extnUuid)
