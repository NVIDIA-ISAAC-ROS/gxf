'''
 SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
import argparse
import gxf.core as core
import sys
import traceback


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='GXE application')
    parser.add_argument('--app', type=str, default='gxf/python/gxe/test_core.yaml',
                        help='GXF app file to execute')
    parser.add_argument('--manifest', type=str, nargs='+', default=['gxf/python/gxe/gxe_manifest.yaml'],
                        help='GXF manifest file with extensions. Multiple files can be comma-separated')
    parser.add_argument('--severity', default=3, type=int,
                        help='Set log severity levels: 0=None, 1=Error, 2=Warning, 3=Info, 4=Debug. Default: Info')
    parser.add_argument('--log_file', type=str,
                        help='Path of a file for logging')
    return parser.parse_args(args)


def main():

    args = parse_args()
    print("creating context...")
    context = core.context_create()
    assert(context is not None)
    try :
        print("loading extensions...")
        core.load_extensions(context, manifest_filenames=args.manifest)
        print("loading graph file...")
        core.graph_load_file(context, args.app)
        print("activating graph...")
        core.graph_activate(context)
        print("running the graph...")
        core.graph_run(context)
        print("running completed successfully. deactivating graph...")
        core.graph_deactivate(context)
        print("destroying context...")
        core.context_destroy(context)
        print("done.")
    except ValueError as valerr:
        print ("Value error:\n")
        print(valerr)
    except Exception as e:
        print("Unknown error:\n")
        print(traceback.format_exc() + e.__str__)


if __name__ == "__main__":
    main()
