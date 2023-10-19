'''
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
"""
This script can be used to generate the boiler plate code required to start a new extension repo

Usage : $ python3 create_extension_repo.py -s sample -n Sample -o '/tmp/sample'
"""

import argparse
import pathlib
import os
import jinja2
import os, os.path as path
from jinja2 import Environment, FileSystemLoader
import shutil
import sys
import uuid

def clean_path(dir_path):
    return path.join(path.abspath(os.path.expanduser(dir_path)),"")

def parse_args(args):
    data = {"extension_short_name": args.short_name[0],
            "extension_name": args.extension_name[0]}
    return data

def main(args):
    data_dict = parse_args(args)

    # Generate UUID
    new_uuid = uuid.uuid4()
    data_dict["full_uuid"] = str(new_uuid)
    data_dict["hash_1"] = f"0x{new_uuid.hex[:16]}"
    data_dict["hash_2"] = f"0x{new_uuid.hex[16:]}"


    env = Environment(
        loader=FileSystemLoader(str(pathlib.Path(__file__).parent/'templates')),
    )

    output_path = clean_path(args.output_path[0])
    templates = env.list_templates()

    for tpl in templates:
        # create templates
        try:
            template = env.get_template(tpl)
        except jinja2.exceptions.TemplateSyntaxError as e:
            print(f"[WARN] Copying template {tpl} without substitution")
            filepath = os.path.join(output_path,tpl)
            if not os.path.exists(str(os.path.dirname(filepath))):
                os.makedirs(str(os.path.dirname(filepath)))
            shutil.copy2("templates/" + tpl,filepath)
            continue

        filename_tpl = env.from_string(tpl)

        # create file content
        content = template.render(data_dict)
        filename = filename_tpl.render(data_dict)

        # Remove tpl extension
        if filename.endswith(".tpl"):
            filename = os.path.splitext(filename)[0]

        # write file content
        filepath = os.path.join(output_path,filename)

        if not os.path.exists(str(os.path.dirname(filepath))):
            os.makedirs(str(os.path.dirname(filepath)))

        with open(filepath, 'w') as f:
            f.write(content)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--short_name",
                        help="Extension short name. Example: sample",
                        required=True,
                        nargs=1)
    parser.add_argument("-n",
                        "--extension_name",
                        help="Full extension name. Example: Sample",
                        required=True,
                        nargs=1)
    parser.add_argument("-o",
                        "--output_path",
                        help="Output path when the starter code should be generated. Example: /tmp/sample_ext_repo",
                        required=True,
                        nargs=1)
    args = parser.parse_args()
    if not args.short_name or not args.extension_name or not args.output_path:
        parser.print_help()
        sys.exit(1)

    res = main(args)
    sys.exit(res)
