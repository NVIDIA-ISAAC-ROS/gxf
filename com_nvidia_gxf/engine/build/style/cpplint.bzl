'''
 SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# We only allow certain source files in isaac
_CPP_ALLOWED_EXTENSIONS = ["c", "cpp", "h", "hpp", "tpp"]

# Do not lint auto-generated cap'n'proto or protocol buffer files
_CPP_IGNORED_EXTENSIONS = ["pb.h", "pb.cc", "capnp.h", "capnp.c++"]

# Do not lint auto-generated cap'n'proto or protocol buffer files
_FILTER_OPTIONS = ["-build/c++11", "+build/c++14", "+build/c++17",
                   "-runtime/explicit", "-runtime/references",
                   "-build/pragma_once", "-build/header_guard", "+build/include_alpha"]

# Additional arguments to cpplint.py
_CPPLINT_EXTRA_ARGUMENTS = [
    "--extensions=" + ",".join(_CPP_ALLOWED_EXTENSIONS),
    "--linelength=100",
    "--headers=h,hpp",
    "--filter=" + ",".join(_FILTER_OPTIONS),
]

def _is_source_label(file):
    filename = file
    """ Checks if a label is a valid source """
    for extension in _CPP_IGNORED_EXTENSIONS:
        if filename.endswith("." + extension):
            return False
    for extension in _CPP_ALLOWED_EXTENSIONS:
        if filename.endswith("." + extension):
            return True

    # In the rare case that we need to pass a shared library as a dependeny we ignore it. This
    # is for example currently happening for the yolo package.
    if filename.endswith(".so"): return True

    fail("Unrecognized extension for source file '%s'" % filename)

def _generate_file_locations_impl(ctx):
    paths = []
    for label in ctx.attr.labels:
        file = label.files.to_list()[0]
        if _is_source_label(file.basename):
            paths.append(file.short_path)
    ctx.actions.write(ctx.outputs.file_paths, "\n".join(paths))
    return DefaultInfo(runfiles = ctx.runfiles(files = [ctx.outputs.file_paths]))

_generate_file_locations = rule(
    implementation = _generate_file_locations_impl,
    attrs = { "labels": attr.label_list(allow_files = True) },
    outputs = { "file_paths": "%{name}_files" },
)

def cpplint(name, srcs, tags = []):
    file_locations_label = "_" + name + "_file_locations"
    _generate_file_locations(name = file_locations_label, labels = srcs)

    native.py_test(
        name = "_cpplint_" + name,
        srcs = ["@com_nvidia_gxf//engine/build/style:cpplint"],
        data = srcs + [file_locations_label],
        main = "@com_nvidia_gxf//engine/build/style:cpplint.py",
        args = _CPPLINT_EXTRA_ARGUMENTS + ["$(location %s)" % file_locations_label],
        size = "small",
        tags = ["cpplint", "lint"] + tags,
        python_version = "PY3",
    )
