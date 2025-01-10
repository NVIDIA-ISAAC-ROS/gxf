"""
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_pkg")
load("@com_nvidia_gxf//bzl:docker.bzl", "gxf_docker_image")


def _expand_extension_dep(target):
    """
    Expand the target name into a valid link to a shared library following these rules:
      1) foo               => //gxf/foo:libgxf_foo.so
      2) //gxf/foo     => //gxf/foo:libgxf_foo.so
      3) //gxf/foo:bar => //gxf/foo:libgxf_bar.so
    """

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

    return path + ":libgxf_" + name + ".so"

def _expand_extension_path(target):
    target = str(target)
    target = target.replace(":", "/")
    if target.startswith("@//"):
        target = target[3:]
    if target.startswith("@"):
        target = "external/" + target[1:]
    return target

def _process_extensions(name, extensions):
    ext_sos = [Label(_expand_extension_dep(x)) for x in extensions]
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

def _nice_name(app):
    """ Gives a valid rule name for an app. Ex: my/foo/bar.yaml => my_foo_bar_yaml """
    app = app.replace("/", "_")
    app = app.replace(".", "_")
    return app

def _data_name(app):
    """ Gives a valid data dependency file for an app. Ex: my/foo/bar.yaml => //my/foo:bar.yaml """
    idx = app.rfind("/")
    return "//" + app[:idx] + ":" + app[idx + 1:]

def nv_pygxf_test(
        name,
        app = None,
        manifest_path_hack = None,
        manifest = None,
        extensions = [],
        data = [],
        **kwargs):
    """
    """
    ext_manifest, ext_deps = _process_extensions(name, extensions)
    if manifest == None:
        manifest = ext_manifest
    py_test_data = []
    if app == None:
        py_test_data = data + ext_deps
    else:
        py_test_data = data + [_data_name(app)] + ext_deps
    native.py_test(
        name = name,
        visibility = ["//visibility:public"],
        data = py_test_data,
        **kwargs
    )

    nv_gxf_pkg(
        name = name + "-pkg",
        srcs = [name],
        visibility = ["//visibility:public"],
        testonly = True,
        tags = ["manual"],
    )

# A rule to create a PyGXF app, optionally wrapped in a Docker container
#
#
# When a image target is provided in the docker_base_image argument,
# then in addition to the regular binary and package build, a Docker
# container containing this app will also be built. The name of the
# Docker image target is the same as the app's name with the '-image'
# suffix appended. A '-push' target will be provided that can be used
# by calling it with `bazel run` to push the newly built containers to
# NGC.
#
# The name of the image will be auto-generated from the target
# name. If this is not desired, the docker_name argument can be
# used to override the container name. However, manually specifying
# the name is generally not recommended to avoid naming conflicts.
#
# Similarly, the registry can be changed from the default nvcr.io to a
# custom target using the docker_registry argument. The default
# tagging scheme can be changed using the docker_tag argument.

def nv_pygxf_app(
        name,
        app = None,
        manifest_path_hack = None,
        manifest = None,
        extensions = [],
        data = [],
        docker_base_image = "",
        docker_registry = None,
        docker_repository = None,
        docker_name = None,
        docker_tag = None,
        **kwargs):
    """
    """
    ext_manifest, ext_deps = _process_extensions(name, extensions)
    if manifest == None:
        manifest = ext_manifest

    if app == None:
        py_binary_data = data + ext_deps
    else:
        py_binary_data = data + [_data_name(app)] + ext_deps

    native.py_binary(
        name = name,
        visibility = ["//visibility:public"],
        data = py_binary_data,
        **kwargs
    )

    nv_gxf_pkg(
        name = name + "-pkg",
        srcs = [name],
        visibility = ["//visibility:public"],
        tags = ["manual"],
    )

    gxf_docker_image(
        name = name,
        data = data,
        docker_base_image = docker_base_image,
        docker_registry = docker_registry,
        docker_repository = docker_repository,
        docker_name = docker_name,
        docker_tag = docker_tag,
    )
