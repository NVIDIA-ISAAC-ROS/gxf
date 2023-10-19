################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import os, os.path as path
import sys
import yaml

class YamlLoader:

    @staticmethod
    def _is_text_file(fp):
        # gathering text elements from ascii table
        textchars = bytearray([7, 8, 9, 10, 12, 13, 27]) + bytearray(range(0x20, 0x7f)) + bytearray(range(0x80, 0x100))

        def is_text(bytes):
            return not bool(bytes.translate(None, textchars))

        with open(fp, 'rb') as f:
            text_sample = f.read(1024)
            return is_text(text_sample)

    def load_yaml(self, file_path: str):
        """ Loads a file with a single yaml document and returns the contents
        """
        if not path.isfile(file_path):
            print(f"Missing file {file_path}")
            return None
        try:
            if not self._is_text_file(file_path):
                print("Yaml Loader: Invalid file, not a text file")
                return None
            with open(file_path) as fp:
                doc = copy.deepcopy(yaml.safe_load(fp))
                return doc
        except yaml.YAMLError as exc:
            if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                print("Yaml Loader: Invalid yaml file. Error position: (%s:%s)" % (mark.line+1, mark.column+1))
                return None

    def load_all(self, file_path: str):
        """ Loads a file with a multiple yaml documents and returns the contents
        """
        if not path.isfile(file_path):
            print(f"Missing file {file_path}")
            return None
        try:
            with open(file_path) as fp:
                docs = copy.deepcopy(list(yaml.load_all(fp, Loader=yaml.SafeLoader)))
                return docs
        except yaml.YAMLError as exc:
             if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                print("Yaml Loader: Invalid yaml file. Error position: (%s:%s)" % (mark.line+1, mark.column+1))
                return None

    def load_string(self, yaml_string):
        """ Loads a string with a single yaml document and returns the contents
        """

        try:
            doc = yaml.safe_load(yaml_string)
            return doc
        except yaml.YAMLError as exc:
            if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                print("Yaml Loader: Invalid yaml file. Error position: (%s:%s)" % (mark.line+1, mark.column+1))
                return None
