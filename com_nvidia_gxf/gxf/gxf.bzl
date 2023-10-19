"""
 SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

load("//engine/build/style:cpplint.bzl", "cpplint")
load("//coverity/bazel:coverity.bzl", "coverity")


_gxf_tag = "gxf"

def _has_code(srcs, hdrs = []):
    return len(srcs + hdrs) > 0

def _shall_lint(tags):
    return tags == None or "nolint" not in tags

def _shall_run_coverity(tags):
    return tags == None or "nocoverity" not in tags

def nv_gxf_c_library(name, srcs = [], hdrs = [], tags = [], **kwargs):
    """
    A standard c_library.
    """

    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        tags = tags + [_gxf_tag],
        **kwargs
    )

    if _has_code(srcs, hdrs):
        if _shall_lint(tags):
            cpplint(name = name, srcs = srcs + hdrs, tags = [_gxf_tag])

        if _shall_run_coverity(tags):
            coverity(name = name, tags = tags)

def nv_gxf_cc_binary(name, srcs = [], tags = [], **kwargs):
    """
    A standard cc_binary.
    """

    native.cc_binary(
        name = name,
        srcs = srcs,
        tags = tags + [_gxf_tag],
        **kwargs
    )

    if _has_code(srcs):
        if _shall_lint(tags):
            cpplint(name = name, srcs = srcs, tags = [_gxf_tag])

        if _shall_run_coverity(tags):
            coverity(name = name, tags = tags)

def nv_gxf_cc_library(
        name,
        alwayslink = True,
        need_multithread = False,
        srcs = [],
        hdrs = [],
        tags = [],
        linkopts = [],
        **kwargs):
    """
    A standard cc_library.
    """
    if need_multithread == True:
        native.cc_library(
            name = name,
            srcs = srcs,
            alwayslink = alwayslink,
            hdrs = hdrs,
            linkopts = linkopts + ["-pthread"],
            tags = tags + [_gxf_tag],
            **kwargs
        )
    else:
        native.cc_library(
            name = name,
            srcs = srcs,
            alwayslink = alwayslink,
            hdrs = hdrs,
            linkopts = linkopts,
            tags = tags + [_gxf_tag],
            **kwargs
        )

    if _has_code(srcs, hdrs):

        if _shall_lint(tags):
            cpplint(name = name, srcs = srcs + hdrs, tags = [_gxf_tag])

        if _shall_run_coverity(tags):
            coverity(name = name, tags = tags)


# Provider to store direct dependencies of an extension
DependencyInfo = provider(
    fields = {
        "ext_deps": "Extension library targets"
    },
)

# Provider to store direct and indirect dependencies of an extension
# in a depset
BaseExtensions = provider(
    fields = {
        "base_exts": "Extension library targets"
    },
)

# Provider to store graph targets
GraphCollector = provider(
    fields = {
              "graphs": "Graph application files",
              "extensions": "Extension library targets",
              "data": "Data files / targets",
             },
)

def _gxf_ext_deps_aspect_impl(target, ctx):
    deps = depset()
    for dep in ctx.rule.attr.ext_deps:
        deps = depset(transitive=[dep[DependencyInfo].ext_deps, deps])

    return [DependencyInfo(ext_deps = deps)]

gxf_ext_deps_aspect = aspect(
    implementation = _gxf_ext_deps_aspect_impl,
    attr_aspects = ["ext_deps"],
)

def _gxf_ext_deps_impl(ctx):
    deps = depset([ctx.attr.ext])
    for dep in ctx.attr.ext_deps:
        deps = depset(transitive=[dep[BaseExtensions].base_exts, deps])

    file_list = _get_files_from_target(deps.to_list())
    rfiles = ctx.runfiles(file_list)
    return [BaseExtensions(base_exts = deps),
            DefaultInfo(files = depset(file_list), runfiles=rfiles)]

nv_gxf_ext_deps = rule(
    implementation = _gxf_ext_deps_impl,
    executable = False,
    attrs = {
        "ext": attr.label(
            allow_single_file = [".so"],
            mandatory = True),
        "ext_deps": attr.label_list(
            doc = "Required dependent extensions",
            aspects = [gxf_ext_deps_aspect],),
        },
)


def nv_gxf_cc_extension(
        name,
        interfaces = [],
        srcs = [],
        hdrs = [],
        visibility = None,
        deps = [],
        data = [],
        ext_deps = [],
        **kwargs):
    """
    Creates an GXF extension as DSO with file name libgxf_XXX.so
    where XXX is the desired name of the module.
    """

    # default_deps = []#[Label("//engine/alice"), Label("//engine/core"), Label("//messages")]
    # for x in default_deps:
    #     if x in deps:
    #         deps.remove(x)
    # deps = deps + default_deps

    interface_lib_name = name
    source_lib_name = name + "_src"

    kwargs["linkopts"] = kwargs.get("linkopts", []) + ["-Wl,-rpath,$$ORIGIN"]

    if visibility == None:
        visibility = ["//visibility:public"]

    nv_gxf_cc_library(
        name = interface_lib_name,
        visibility = visibility,
        hdrs = hdrs + interfaces,
        deps = deps + ext_deps,
        **kwargs
    )

    nv_gxf_cc_library(
        name = source_lib_name,
        visibility = visibility,
        srcs = srcs,
        hdrs = hdrs + interfaces,
        deps = deps + [interface_lib_name],
        **kwargs
    )

    base_exts = [Label(_expand_extension(x)) for x in ext_deps]
    ext_dep_labels = [Label(_expand_extension_dep(x)) for x in ext_deps]

    # Create the shared library for the module
    native.cc_binary(
        name = "libgxf_" + name + ".so",
        visibility = visibility,
        deps = deps + [source_lib_name],
        data = base_exts + data,
        linkshared = True,
        **kwargs
    )

    # Create target depset of extension + direct + indirect dependencies
    nv_gxf_ext_deps(name = name + "_ext_deps",
                    ext = "libgxf_" + name + ".so",
                    ext_deps = ext_dep_labels,
                    visibility = visibility)

def _gxf_get_app_deps_path(deps):
    paths = {}
    libs_string = ""
    for i, d in enumerate(deps):
        for f in d.data_runfiles.files.to_list():
            # Only include source files
            if (f.dirname.rfind("bazel-out") < 0):
                paths[f.dirname] = True

        libs_string = ":".join(paths.keys())
    return (libs_string)

def _gxf_app_runscript_impl(ctx):
    lib_path = _gxf_get_app_deps_path(ctx.attr.deps)

    graph_files = ctx.files.app_yaml_files

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = {
            "{APP_YAML_FILE}": ",".join([f.short_path for f in graph_files]),
            "{MANIFEST_YAML_FILE}": ctx.file.manifest.short_path,
            "{LIB_PATH}": lib_path,
        },
    )

_gxf_app_runscript = rule(
    implementation = _gxf_app_runscript_impl,
    output_to_genfiles = True,
    attrs = {
        "script_name": attr.string(mandatory = True),
        "manifest": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "app_yaml_files": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "sub_graphs": attr.label_list(
            providers = [GraphCollector],
            mandatory = True,
            allow_files = True,
        ),
        "template": attr.label(
            default = Label("@com_nvidia_gxf//gxf:run.sh.tpl"),
            allow_single_file = True,
        ),
        "deps": attr.label_list(
            allow_files = True,
        )
    },
    outputs = {"out": "run_%{script_name}"},
)

def _expand_path(target):
    # Split by : to get path and target name
    tokens = target.split(":")
    if len(tokens) == 1:
        path = target
        name = target
        has_colon = False
    elif len(tokens) == 2:
        path = tokens[0]
        name = tokens[1]
        has_colon = True
    else:
        print("Invalid token:", target)
        return ""

    # Split path by '/' to get full path
    tokens = path.split("/")
    if len(tokens) == 1:
        path = "//" + native.package_name()
    if not has_colon:
        name = tokens[-1]

    if path.startswith("//"):
        path = "@" + native.repository_name() + path

    return path, name

def _expand_extension(target):
    """
    Expand the target name into a valid link to a shared library following these rules:
      1) foo               => //path/to/foo:libgxf_foo.so
      2) //gxf/foo     => //gxf/foo:libgxf_foo.so
      3) //gxf/foo:bar => //gxf/foo:libgxf_bar.so
    """

    path , name = _expand_path(target)
    return path + ":libgxf_" + name + ".so"

def _expand_extension_dep(target):
    """
    Expand the target name into a valid link to a shared library following these rules:
      1) foo               => //path/to/foo:foo_ext_deps
      2) //gxf/foo     => //gxf/foo:foo_ext_deps
      3) //gxf/foo:bar => //gxf/foo:bar_ext_deps
    """

    path , name = _expand_path(target)
    return path + ":" + name + "_ext_deps"

def _expand_sub_graph_files(target):
    """
    Expand the target name into a valid link to a shared library following these rules:
      1) foo               => //path/to/foo:foo_files
      2) //gxf/foo     => //gxf/foo:foo_ext_files
      3) //gxf/foo:bar => //gxf/foo:bar_ext_files
    """

    path , name = _expand_path(target)
    return path + ":" + name + "_files"

def _expand_extension_path(target):
    target = str(target)
    target = target.replace(":", "/")
    if target.startswith("@//"):
        target = target[3:]
    if target.startswith("@"):
        target = "external/" + target[1:]
    return target


def _process_extensions(name, extensions):
    ext_sos = [Label(_expand_extension(x)) for x in extensions]
    ext_paths = [_expand_extension_path(x) for x in ext_sos]

    manifest_str = "extensions:\n"
    for path in ext_paths:
        manifest_str += "- {}\n".format(path)

    manifest_target = "{}_manifest".format(name)
    manifest_filename = "{}_manifest.yaml".format(name)

    native.genrule(
        name = manifest_target,
        outs = [manifest_filename],
        cmd = "cat <<'EOF' >$@\n" + manifest_str,
    )

    return [manifest_filename, [manifest_filename] + ext_sos]


def _gxf_pkg_impl(ctx):
    arg_compression = ""

    # Analyse desired compression
    if ctx.attr.extension:
        dotPos = ctx.attr.extension.find(".")
        if dotPos > 0:
            dotPos += 1
            suffix = ctx.attr.extension[dotPos:]
            if suffix == "gz":
                arg_compression = "--gzip"
            elif suffix == "bz2":
                arg_compression = "--bzip2"
            else:
                fail("Unsupported compression '%s'" % ctx.attr.extension)

    files = depset()

    # Collect all datafiles of all data
    for dep in ctx.attr.data:
        if hasattr(dep, "data_runfiles"):
            files = depset(transitive = [files, dep.data_runfiles.files])

    # Collect all runfiles of all dependencies
    for dep in ctx.attr.srcs:
        if hasattr(dep, "default_runfiles"):
            files = depset(transitive = [files, dep.default_runfiles.files])
    files = files.to_list()

    exc_files = depset()
    for dep in ctx.attr.excludes:
        if hasattr(dep, "default_runfiles"):
            exc_files = depset(transitive = [exc_files, dep.default_runfiles.files])
    exc_files = exc_files.to_list()

    for f in exc_files:
        if not f.is_source and f in files:
            files.remove(f)

    # Create a space-separate string with paths to all files
    file_list = " ".join([f.path for f in files])

    # Setup a rule to move files from bazel-out to the root folder
    bazel_out_rename = "--transform='flags=r;s|bazel-out/k8-opt/bin/||' " + \
                       "--transform='flags=r;s|bazel-out/k8-fastbuild/bin/||' " + \
                       "--transform='flags=r;s|bazel-out/aarch64-opt/bin/||' " + \
                       "--transform='flags=r;s|bazel-out/aarch64_sbsa-opt/bin/||' " + \
                       "--transform='flags=r;s|bazel-out/aarch64-fastbuild/bin/||' " + \
                       "--transform='flags=r;s|" + ctx.attr.strip_prefix + "||' "

    # Additional replacement rules
    for key, value in ctx.attr.remap_paths.items():
        bazel_out_rename += "--transform='flags=r;s|%s|%s|' " % (key, value)

    # Create the tar archive
    ctx.actions.run_shell(
        command = "tar --hard-dereference %s %s  -chf %s %s" %
                  (arg_compression, bazel_out_rename, ctx.outputs.out.path, file_list),
        inputs = files,
        outputs = [ctx.outputs.out],
        use_default_shell_env = True,
    )

# A rule which creates a tar package with the executable and all necessary runfiles compared to
# pkg_tar which needs manual dependency tracing.
_nv_gxf_pkg = rule(
    implementation = _gxf_pkg_impl,
    executable = False,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "data": attr.label_list(allow_files = True),
        "extension": attr.string(default = "tar"),
        "strip_prefix": attr.string(default = ""),
        "remap_paths": attr.string_dict(),
        "excludes": attr.label_list(allow_files = True),
    },
    outputs = {
        "out": "%{name}.%{extension}",
    },
)

def nv_gxf_pkg(name, srcs, data = [], remap_paths = {}, excludes = [], **kwargs):
    '''
    Creates a package containing all targets specified in `srcs`. `nv_gxf_pkg` tracks transitive
    dependencies automatically and thus will include all necessary runfiles. The package also
    includes a script "run" which can be used to conveniently run applications.
    '''
    _nv_gxf_pkg(
        name = name,
        srcs = srcs,
        data = data,
        remap_paths = remap_paths,
        excludes = excludes,
        **kwargs
    )

def _gxf_graph_impl(ctx):
    ext_bin = depset()
    for x in ctx.attr.ext_deps:
        ext_bin = depset(transitive = [ext_bin, x[BaseExtensions].base_exts])

    return [GraphCollector(
                        graphs = ctx.attr.sub_graphs,
                        extensions = ext_bin,
                        data = ctx.attr.data,
                        )
            ]

_nv_gxf_sub_graph = rule(
    implementation = _gxf_graph_impl,
    executable = False,
    attrs = {
        "sub_graphs": attr.label_list(
                allow_files = True,
                mandatory = True,
                doc = "Application graph files"),
        "ext_deps": attr.label_list(
                allow_files = True,
                mandatory = True,
                doc = "Extension library targets"),
        "data": attr.label_list(
                allow_files = True,
                doc = "Data files / targets"),
    },
)

# A rule which creates a graph target that can be consumed by nv_gxf_app targets
# These targets would be useful to create reuseable sub graphs
# Note: This rule does not execute the graph itself.

def nv_gxf_sub_graph(
    name,
    sub_graphs = [],
    data = [],
    extensions = []):

    # Collect all extension depdencies
    # //gxf/foo:bar => //gxf/foo:foo_ext_deps
    ext_dep_labels = [Label(_expand_extension_dep(x)) for x in extensions]
    ext_labels = [Label(_expand_extension(x)) for x in extensions]

    _nv_gxf_sub_graph(
        name = name,
        sub_graphs = sub_graphs,
        data = data,
        ext_deps = ext_dep_labels,
        visibility = ["//visibility:public"]
    )

    # Create filegroup target of extensions and graph files
    native.filegroup(
        name = name + "_files",
        data = sub_graphs + ext_labels + ext_dep_labels + data,
        visibility = ["//visibility:public"],
    )

def nv_gxf_test_app(
        name,
        srcs = [],
        manifest = None,
        extensions = [],
        data = [],
        deps = [],
        tags = [],
        sub_graphs = [],
        **kwargs):

    manifest_target = "{}_manifest".format(name)
    manifest_filename = "{}_manifest.yaml".format(name)
    ext_labels = [Label(_expand_extension(x)) for x in extensions]

    # Prepare list of base extension (direct dependencies) depset targets for all extensions
    all_ext_deps = []
    for x in ext_labels:
        ext_name = x.name.split(".so")[0].split("libgxf_")[1]
        all_ext_deps.append(x.relative(ext_name + "_ext_deps"))

    # Merge all extensions + merged subgraph list to create the finall app manifest
    sub_graph_files = [Label(_expand_sub_graph_files(x)) for x in sub_graphs]
    nv_gxf_cc_prepare_extensions(manifest_target, manifest_filename, all_ext_deps, sub_graphs)
    ext_deps = [manifest_filename] + ext_labels + sub_graph_files

    if manifest == None:
        manifest = manifest_filename

    script = "_script_" + name
    _gxf_app_runscript(
        name = script,
        script_name = name,
        app_yaml_files = srcs,
        manifest = manifest,
        sub_graphs = sub_graphs,
        deps = deps,
    )

    native.sh_test(
        tags = tags,
        name = name,
        srcs = [script],
        data = [
            Label("@com_nvidia_gxf//gxf/gxe"),
        ] + srcs + ext_deps + data,
        **kwargs
    )

    nv_gxf_pkg(
        name = name + "-pkg",
        srcs = [name],
        visibility = ["//visibility:public"],
        tags = ["manual"],
        testonly = True,
    )

def nv_gxf_app(
        name,
        gxf_file = None,  # deprecated -- use srcs instead
        srcs = [],
        manifest = None,
        extensions = [],
        data = [],
        deps = [],
        tags = [],
        sub_graphs = [],
        **kwargs):
    """
    Defines a default GXF application. The application depends on a couple of extensions and is
    started via an app yaml file. `extensions` gives an easy way to specify modules which are found
    in the //gxf folder.
    """

    manifest_target = "{}_manifest".format(name)
    manifest_filename = "{}_manifest.yaml".format(name)
    ext_labels = [Label(_expand_extension(x)) for x in extensions]

    # Prepare list of base extension (direct dependencies) depset targets for all extensions
    all_ext_deps = []
    for x in ext_labels:
        ext_name = x.name.split(".so")[0].split("libgxf_")[1]
        all_ext_deps.append(x.relative(ext_name + "_ext_deps"))

    # Merge all extensions + merged subgraph list to create the finall app manifest
    sub_graph_files = [Label(_expand_sub_graph_files(x)) for x in sub_graphs]
    nv_gxf_cc_prepare_extensions(manifest_target, manifest_filename, all_ext_deps, sub_graphs)
    ext_deps = [manifest_filename] + ext_labels + sub_graph_files

    if manifest == None:
        manifest = manifest_filename

    if gxf_file == None:
        srcs_tmp = srcs
    else:
        srcs_tmp = srcs + [gxf_file]

    script = "_script_" + name
    _gxf_app_runscript(
        name = script,
        script_name = name,
        app_yaml_files = srcs_tmp,
        manifest = manifest,
        deps = deps,
        sub_graphs = sub_graphs
    )

    native.sh_binary(
        tags = tags,
        name = name,
        srcs = [script],
        data = [
            Label("@com_nvidia_gxf//gxf/gxe"),
        ] + srcs_tmp + ext_deps + data + deps,
        **kwargs
    )

    nv_gxf_pkg(
        name = name + "-pkg",
        srcs = [name],
        visibility = ["//visibility:public"],
        tags = ["manual"],
    )

    if _shall_lint(tags):
        # Pop args from kwargs because args is already defined for the sh_test below.
        kwargs.pop('args', None)
        native.sh_test(
            name = "_gxflint_" + name,
            tags = ["gxflint", "lint"] + tags,
            srcs = [script],
            data = [
                Label("@com_nvidia_gxf//gxf/gxe"),
            ] + srcs_tmp + ext_deps + data + deps,
            args = ["--run=false"],
            **kwargs
        )

def _nice_name(app):
    """ Gives a valid rule name for an app. Ex: my/foo/bar.yaml => my_foo_bar_yaml """
    app = app.replace("/", "_")
    app = app.replace(".", "_")
    return app

def _data_name(app):
    """ Gives a valid data dependency file for an app. Ex: my/foo/bar.yaml => //my/foo:bar.yaml """
    idx = app.rfind("/")
    return "//" + app[:idx] + ":" + app[idx + 1:]

def get_files_path_from_file_list(fl):
    paths = []
    for dep in fl:
        size_prefix_remove = len(dep.root.path)
        if size_prefix_remove != 0:
            size_prefix_remove += 1
        paths.append(dep.path[size_prefix_remove:])
    return paths

def _get_files_from_target(tgt):
    all = []
    for dep in tgt:
        all += dep.files.to_list()
    all = depset(all).to_list()
    return all

def _make_manifest_yaml(ctx, name, extensions):
    manifest_str = "extensions:\n"
    for path in extensions:
        manifest_str += "- {}\n".format(path)

    file_manifest = ctx.actions.declare_file(name)
    ctx.actions.write(file_manifest, manifest_str)
    return file_manifest

FileCollector = provider(
                            fields = {
                                "file_out": "output file",
                                "extensions": "extension targets"
                            }
                        )

def _nv_gxf_cc_prepare_extensions_impl(ctx):
    """
        Creates one cc_test target per GXF app file given in `apps`. The test is given the same name as
        the corresponding app file. The test app is passed as an argument via "--app".
        """

    ext_bin = depset()
    for x in ctx.attr.extensions:
        ext_bin = depset(transitive = [ext_bin, x[BaseExtensions].base_exts])

    sub_graph_extensions = depset()
    for x in ctx.attr.sub_graphs:
        sub_graph_extensions = depset(transitive = [sub_graph_extensions, x[GraphCollector].extensions])

    ext_bin = depset(transitive = [ext_bin, sub_graph_extensions])
    extension_files = _get_files_from_target(ext_bin.to_list())
    extensions = get_files_path_from_file_list(extension_files)
    file_manifest = _make_manifest_yaml(ctx, ctx.attr.manifest_filename, extensions)

    file_list = _get_files_from_target(ext_bin.to_list())
    rfiles = ctx.runfiles(file_list)
    return [FileCollector(file_out = file_manifest, extensions = ext_bin.to_list()),
            DefaultInfo(files = depset(file_list), runfiles=rfiles)]



nv_gxf_cc_test_group_rule = rule(
    implementation=_nv_gxf_cc_prepare_extensions_impl,
    attrs=
    {
        "manifest_filename": attr.string(
            doc="manifest required for packages"),
        "extensions": attr.label_list(
            providers = [BaseExtensions],
            doc="extension deps targets required for packages",
            allow_files=True),
        "manifest_out": attr.output(
            doc = "extension metadata output",
            mandatory = True,),
        "sub_graphs": attr.label_list(
            providers = [GraphCollector],
            doc="sub graphs required for packages"),
    })


def nv_gxf_cc_prepare_extensions(manifest_target, manifest_filename, extensions=[], sub_graphs=[]):
    nv_gxf_cc_test_group_rule(
        name=manifest_target,
        manifest_filename=manifest_filename,
        extensions=extensions,
        sub_graphs = sub_graphs,
        manifest_out = manifest_filename,
        visibility = ["//visibility:public"],
    )

def nv_gxf_cc_test_group(name, apps, manifest_path_hack, manifest = None, extensions = [],
                         data = [], use_app_as_name = True, **kwargs):
    """
    Creates one cc_test target per GXF app file given in `apps`. The test is given the same name as
    the corresponding app file. The test app is passed as an argument via "--app".
    """
    manifest_target = "{}_manifest".format(name)
    manifest_filename = "{}_manifest.yaml".format(name)
    ext_labels = [Label(_expand_extension(x)) for x in extensions]

    # Prepare list of base extension (direct dependencies) depset targets for all extensions
    all_ext_deps = []
    for x in ext_labels:
        ext_name = x.name.split(".so")[0].split("libgxf_")[1]
        all_ext_deps.append(x.relative(ext_name + "_ext_deps"))

    nv_gxf_cc_prepare_extensions(manifest_target, manifest_filename, all_ext_deps)
    ext_deps = [manifest_filename] + ext_labels

    if manifest == None:
        manifest = manifest_filename

    all_tests = []
    all_tests_commands = []
    for app in apps:
        app_name = _nice_name(app) if use_app_as_name else (_nice_name(name) + "_yaml")
        args = ["--app", app, "--manifest", manifest_path_hack + manifest]
        full_command = " ".join([manifest_path_hack + app_name] + args)
        all_tests_commands.append(full_command)
        native.cc_test(
            name = app_name,
            visibility = ["//visibility:public"],
            args = args,
            data = data + [_data_name(x) for x in app.split(",")] + ext_deps,
            **kwargs
        )
        all_tests.append(app_name)

    nv_gxf_pkg(
        name = name,
        srcs = all_tests,
        testonly = True,
        tags = ["manual"],
    )

def _components_pyclass_gen_internal(ctx):
    if ctx.attr.skip:
        components_py = ctx.actions.declare_file("Components.py")
        ctx.actions.write(components_py, content = "")
        return DefaultInfo(
                files = depset([components_py]))

    ext_bin = depset()
    for x in ctx.attr.extensions:
        ext_bin = depset(transitive = [ext_bin, x[BaseExtensions].base_exts])

    # Collect all extension runfiles
    e_runfiles = []
    for e in ext_bin.to_list():
        e_runfiles += e[DefaultInfo].data_runfiles.files.to_list()
        e_runfiles += e[DefaultInfo].default_runfiles.files.to_list()

    extension_files = _get_files_from_target(ext_bin.to_list())
    extensions = get_files_path_from_file_list(extension_files)
    args = ctx.actions.args()
    args.add("--bazel_workspace_root", ctx.label.workspace_root)
    args.add("--bazel_workspace",  ctx.bin_dir.path)
    args.add_all("--libraries", extensions)
    ctx.actions.run(
        outputs = ctx.outputs.outs,
        inputs = extension_files + e_runfiles,
        arguments = [args],
        executable = ctx.executable.tool,
        mnemonic = "RunBinary",
    )

    return DefaultInfo(
                files = depset(ctx.outputs.outs),
                runfiles = ctx.runfiles(ctx.outputs.outs + extension_files + e_runfiles))


nv_gxf_components_pyclass_gen_internal = rule(
        implementation = _components_pyclass_gen_internal,
        attrs = {
            "tool": attr.label(
                doc = "The tool to run in the action.<br/><br/>Must be the label of a *_binary rule," +
                    " of a rule that generates an executable file, or of a file that can be" +
                    " executed as a subprocess (e.g. an .exe or .bat file on Windows or a binary" +
                    " with executable permission on Linux). This label is available for" +
                    " <code>$(location)</code> expansion in <code>args</code> and <code>env</code>.",
                executable = True,
                allow_files = True,
                mandatory = True,
                cfg = "host",
            ),
            "extensions": attr.label_list(
                providers = [BaseExtensions],
                allow_files = True,
                mandatory = True,
                doc = "Additional inputs of the action.<br/><br/>These labels are available for" +
                    " <code>$(location)</code> expansion in <code>args</code> and <code>env</code>.",
            ),
            "outs": attr.output_list(
                mandatory = True,
                doc = "Output files generated by the action.<br/><br/>These labels are available for" +
                    " <code>$(location)</code> expansion in <code>args</code> and <code>env</code>.",
            ),
            "skip": attr.bool(
                doc = "Choose to skip pymodule generation",
                default = True),
        },
    )

def nv_gxf_components_pyclass_gen(name, extensions, visibility = None):
    ext_labels = [Label(_expand_extension(x)) for x in extensions]

    # Prepare list of base extension (direct dependencies) depset targets for all extensions
    all_ext_deps = []
    for x in ext_labels:
        ext_name = x.name.split(".so")[0].split("libgxf_")[1]
        all_ext_deps.append(x.relative(ext_name + "_ext_deps"))

    nv_gxf_components_pyclass_gen_internal(
        name=name,
        tool = "@com_nvidia_gxf//gxf/core:py_module_builder",
        extensions = all_ext_deps,
        outs = ["Components.py"],
        visibility = visibility,
        skip = select({
            "@com_nvidia_gxf//engine/build:cpu_host": False,
            "//conditions:default": True})
            )
