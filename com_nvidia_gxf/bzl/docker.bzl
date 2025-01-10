"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

load("@io_bazel_rules_docker//lang:image.bzl", "app_layer")
load("@io_bazel_rules_docker//container:container.bzl", "container_image", "container_push")
load("@com_nvidia_gxf//bzl:utils.bzl", "label_workspace_relative_path", "package_relative_label")

def default_container_registry():
    return "nvcr.io"

def default_container_repository():
    return "nvstaging/osmo"

def default_container_name(app_label_string):
    """
    Get the default name for a container given the corresponding app label string.

    Example: //foo/bar:baz -> foo_bar_baz
            :baz           -> foo_bar_baz
    """
    app_label = package_relative_label(app_label_string)
    app_path = label_workspace_relative_path(app_label)
    container_name = "".join([c if c.isalnum() else "_" for c in app_path.elems()])
    return container_name

def default_container_tag():
    # The tag is expected to be substituted with build args and bazel's voltile status.
    # Example substituted tag: `master-k8`
    container_tag = "{BUILD_BRANCH_TAG}-$(TARGET_CPU)"
    return container_tag

def default_container_full_name(app_label_string):
    # Example full name: nvcr.io/nvstaging/osmo/my_app:master-k8
    return "{}/{}/{}:{}".format(
        default_container_registry(),
        default_container_repository(),
        default_container_name(app_label_string),
        default_container_tag(),
    )

def gxf_docker_push(
        name,
        docker_base_image,
        tags = [],
        docker_registry = None,
        docker_repository = None,
        docker_name = None,
        docker_tag = None,
        stamp = "@io_bazel_rules_docker//stamp:always"):
    """
    Helper to push a docker image to a registry for a given app.

    Per default we set the stamp to `always` because the default docker_tag relies on the stamp.
    """

    # Resolve default arguments:
    docker_registry = docker_registry if docker_registry != None else default_container_registry()
    docker_repository = docker_repository if docker_repository != None else default_container_repository()
    docker_name = docker_name if docker_name != None else default_container_name(name)
    container_path = docker_repository + "/" + docker_name
    docker_tag = docker_tag if docker_tag != None else default_container_tag()

    container_push(
        name = name + "-push",
        image = docker_base_image,
        format = "Docker",
        registry = docker_registry,
        repository = container_path,
        visibility = ["//visibility:public"],
        tag = docker_tag,
        tags = ["docker"] + tags,
        stamp = stamp,
    )

def gxf_docker_image(
        name,
        docker_base_image,
        data = [],
        tags = [],
        entrypoint = [],
        docker_registry = None,
        docker_repository = None,
        docker_name = None,
        docker_tag = None,
        create_empty_workspace_dir = False,
        stamp = "@io_bazel_rules_docker//stamp:always"):
    """
    Helper to generate docker targets for a given app.

    Per default we set the stamp to `always` because the default docker_tag relies on the stamp.
    """
    if docker_base_image != "":
        architecture = select(
            {
                "@com_nvidia_gxf//engine/build:cpu_host": "amd64",
                "@com_nvidia_gxf//engine/build:cpu_aarch64": "arm64",
                "@com_nvidia_gxf//conditions:default": "amd64",
            },
            no_match_error = "Please build only for x86 or arm64 platforms only",
        )

        docker_versioned_image_name = "_versioned_" + name + "-image"
        docker_image_name = name + "-image"
        app_layer(
            name = docker_image_name,
            base = docker_base_image,
            binary = name,
            visibility = ["//visibility:public"],
            data = data,
            tags = ["docker"] + tags,
            entrypoint = entrypoint,
            create_empty_workspace_dir = create_empty_workspace_dir,
            architecture = architecture,
        )

        gxf_docker_push(
            name,
            docker_image_name,
            tags = tags,
            docker_registry = docker_registry,
            docker_repository = docker_repository,
            docker_name = docker_name,
            docker_tag = docker_tag,
            stamp = stamp,
        )

        # Disable versioned image push until CI support has been enabled
        # container_image(
        #     name = docker_versioned_image_name,
        #     base = name + "-image",
        #     layers = ["//bzl:version_txt_layer"],
        #     tags = ["docker"] + tags,
        #     architecture = architecture,
        # )

        # gxf_docker_push(
        #     name,
        #     docker_versioned_image_name,
        #     tags = tags,
        #     docker_registry = docker_registry,
        #     docker_repository = docker_repository,
        #     docker_name = docker_name,
        #     docker_tag = docker_tag,
        #     stamp = stamp,
        # )
