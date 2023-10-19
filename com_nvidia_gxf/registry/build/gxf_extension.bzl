"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "patch", "workspace_and_buildfile")

LogCollector = provider(
    fields = {"interface": "extension interface publish log",
              "variant": "extension variant publish log"}
)

def _get_registry_cmd(repository_ctx):
    # Get registry cmd and its path
    # If registry is not installed, return empty string

    exec_res = repository_ctx.execute([
        "which",
        "registry"
    ])
    if exec_res.return_code != 0:
        print("Failed to search preinstalled registry binary:")
        return ("")
    else:
        print("Registry binary found at " + exec_res.stdout)

    registry_path = exec_res.stdout.rstrip()
    exec_res = repository_ctx.execute([registry_path, "-v"])
    if exec_res.return_code == 0:
        return (registry_path)
    else:
        return ("")

def _prepare_registry_script(repository_ctx):
    # Install registry if has not been installed
    registry_cmd = _get_registry_cmd(repository_ctx)
    if registry_cmd != "":
        print("Registry CLI is alreay installed")
        return (registry_cmd)
    tar_file = "registry_binary-any-any-release-23.04_20230420_71698e3f_internal.tar"
    exec_res = repository_ctx.execute([
        "wget",
        "https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/nightly/release-23.04/" +
        tar_file,
    ])
    if exec_res.return_code != 0:
        fail("Failed to download registry tar file" + exec_res.stdout)

    exec_res = repository_ctx.execute([
        "tar",
        "-xvf",
        tar_file,
    ])
    if exec_res.return_code != 0:
        fail("Failed to untar file:" + exec_res.stdout)

    exec_res = repository_ctx.execute(["pwd"])
    print("Registry downloaded to path :" + exec_res.stdout)

    registry_path = exec_res.stdout.rstrip() + "/registry_linux_x86_64"
    exec_res = repository_ctx.execute([registry_path, "-v"])

    if exec_res.return_code == 0:
        print("Registry CLI is downloaded successfully")
        return (registry_path)
    else:
        fail("Failed to install registry")

def _graph_create_repo(repository_ctx, registry_cmd, repo_name):
    if "NGC_API_KEY" in repository_ctx.os.environ:
        api_key = repository_ctx.os.environ["NGC_API_KEY"]
    else:
        print("NGC_API_KEY env var not set. Checking NGC cli config file..")
        exec_res = repository_ctx.execute([
            "cat",
            repository_ctx.os.environ["HOME"] + "/.ngc/config",
        ])
        if exec_res.return_code != 0:
            fail("Can't find ngc configuration under ~/.ngc. Have you installed NGC cli tool ?")

        for line in exec_res.stdout.splitlines():
            if "apikey = " in line:
                api_key = line.strip("apikey =")

    exec_res = repository_ctx.execute([
        registry_cmd,
        "cache",
        "-c",
    ])

    # Check registry repo
    exec_res = repository_ctx.execute([
        registry_cmd,
        "repo",
        "list",
    ])
    if exec_res.return_code != 0:
        fail("Failed to get registry repo list. Error:\n" +
             exec_res.stderr + exec_res.stdout)

    found = False
    for line in exec_res.stdout.splitlines():
        if "- " + repo_name == line:
            found = True
            print("Repo exists...")
    if found == False:
        print("Adding registry repo:" + repo_name)

        # Create repo
        exec_res = repository_ctx.execute([
            registry_cmd,
            "repo",
            "add",
            "ngc",
            "-n",
            repo_name,
            "-a",
            api_key,
            "-o",
            "nv-gxf-dev",
            "-t",
            "ngc-public",
        ])
        if exec_res.return_code != 0:
            print("[WARNING] Failed to add registry repo. Error:\n" +
                 exec_res.stderr + exec_res.stdout+ exec_res.stdout)

    # Sync
    exec_res = repository_ctx.execute([
        registry_cmd,
        "repo",
        "sync",
        "-n",
        repo_name,
    ])
    if exec_res.return_code != 0:
        fail("Failed to sync registry repo, Error:\n" +
             exec_res.stderr + exec_res.stdout)
    print("NGC repo is synced")

def _gxf_import_ext_impl(repository_ctx):
    # Install registry
    registry_cmd = "registry"
    if not repository_ctx.which("registry"):
        registry_cmd = _get_registry_cmd(repository_ctx)
        if registry_cmd == "":
            registry_cmd = _prepare_registry_script(repository_ctx)

    # Create repo
    _graph_create_repo(repository_ctx, registry_cmd, repository_ctx.attr.repo_name)

    # Install extensions
    ext = repository_ctx.attr.ext_name
    exec_res = repository_ctx.execute([
        registry_cmd,
        "extn",
        "info",
        "-n",
        ext,
    ])
    if exec_res.return_code != 0:
        fail("Failed to get extn info for:" + ext + ", Error:\n" +
              exec_res.stderr + exec_res.stdout)
    else:
        print("Extension found - " + ext)
        print(exec_res.stdout)

    for line in exec_res.stdout.splitlines():
        if "uuid" in line:
            ext_uuid = line.split(" ")[-1]

    if ext_uuid == None:
        fail("Failed to get information for extension " + ext +
             ":\n" + exec_res.stderr)

    exec_res = repository_ctx.execute([
        registry_cmd,
        "extn",
        "dependencies",
        "-n",
        ext,
        "-s",
        repository_ctx.attr.version,
    ])
    if exec_res.return_code != 0:
        fail("Failed to get dependencies for extension " + ext +
             "Error:\n" + exec_res.stderr + exec_res.stdout)

    # Import extension
    exec_res = repository_ctx.execute([
        registry_cmd,
        "extn",
        "import",
        "variant",
        "-n",
        ext,
        "-s",
        repository_ctx.attr.version,
        "-a",
        repository_ctx.attr.arch,
        "-f",
        repository_ctx.attr.distribution,
        "-o",
        repository_ctx.attr.os,
        "--cuda",
        repository_ctx.attr.cuda,
        "--vpi",
        repository_ctx.attr.vpi,
        "--tensorrt",
        repository_ctx.attr.tensorrt,
        "-d",
        "./",
    ])
    if exec_res.return_code != 0:
        fail("Failed to import NGC extension " + repository_ctx.attr.name +
             "Error:\n" + exec_res.stderr + exec_res.stdout)

    exec_res = repository_ctx.execute(["ls", ext_uuid + "/"])
    for f in exec_res.stdout.splitlines():
        if "BUILD" in f:
            repository_ctx.execute(["mv", ext_uuid + "/" + f, "BUILD"])
        else:
            repository_ctx.execute(["mv", ext_uuid + "/" + f, f])

    # Add BUILD file
    workspace_and_buildfile(repository_ctx)
    repository_ctx.execute(["mv", "./BUILD.bazel", "./BUILD"])

gxf_import_ext = repository_rule(
    implementation = _gxf_import_ext_impl,
    attrs =
        {
            "ext_name": attr.string(mandatory = True),
            "repo_name": attr.string(mandatory = True),
            "version": attr.string(mandatory = True),
            "arch": attr.string(
                mandatory = True,
                default = "x86_64",
            ),
            "distribution": attr.string(
                mandatory = True,
                default = "ubuntu_20.04",
            ),
            "os": attr.string(
                mandatory = True,
                default = "linux",
            ),
            "cuda": attr.string(
                mandatory = True,
                default = "",
            ),
            "tensorrt": attr.string(
                mandatory = True,
                default = "",
            ),
            "vpi": attr.string(
                mandatory = True,
                default = "",
            ),
            "build_file": attr.label(
                allow_single_file = True,
                mandatory = True,
            ),
            "build_file_content": attr.string(
                doc = "The content for the BUILD file for this repository.",
            ),
            "workspace_file": attr.label(
                doc = "The file to use as the WORKSPACE file for this repository.",
            ),
            "workspace_file_content": attr.string(
                doc = "The content for the WORKSPACE file for this repository.",
            ),
        },
)


def _gxf_publish_ext_impl(ctx):
    # Collect action inputs
    action_inputs = ctx.attr.ext_registration_target.files.to_list()
    action_inputs += ctx.attr._registry_cli_tool[DefaultInfo].data_runfiles.files.to_list()
    action_inputs += ctx.attr._registry_cli_tool[DefaultInfo].default_runfiles.files.to_list()

    # generate registration script
    publish_script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.expand_template(
      template = ctx.file._publish_template,
      output = publish_script,
      substitutions = {
          "{EXTENSION_NAME}": ctx.attr.ext_name,
          "{REPO_NAME}": ctx.attr.repo_name,
          "{ARCH}": ctx.attr.arch,
          "{OS}": ctx.attr.os,
          "{DISTRO}": ctx.attr.distribution,
          "{CUDA}": ctx.attr.cuda,
          "{CUDNN}": ctx.attr.cudnn,
          "{TRITON}": ctx.attr.triton,
          "{TENSORRT}": ctx.attr.tensorrt,
          "{DEEPSTREAM}": ctx.attr.deepstream,
          "{VPI}": ctx.attr.vpi,
          "{FORCE}": str(ctx.attr.force),
      },
      is_executable = True,
    )

    return [
            DefaultInfo(executable = publish_script,
                        runfiles=ctx.runfiles([publish_script] + action_inputs),
                        )
            ]

publish_extension_rule = rule(
    implementation = _gxf_publish_ext_impl,
    executable = True,
    attrs =
        {
            "_registry_cli_tool": attr.label(
                executable = True,
                cfg = "exec",
                allow_files = True,
                default = Label("//registry/cli:registry_cli")),
            "_publish_template": attr.label(
                allow_single_file = [".sh.tpl"],
                default = "//registry/build:publish.sh.tpl"),
            "ext_name": attr.string(mandatory = True),
            "ext_registration_target": attr.label(mandatory = True),
            "repo_name": attr.string(mandatory = True),
            "arch": attr.string(
                mandatory = True,
                default = "x86_64",
            ),
            "distribution": attr.string(
                mandatory = True,
                default = "ubuntu_20.04",
            ),
            "os": attr.string(
                mandatory = True,
                default = "linux",
            ),
            "cuda": attr.string(
                mandatory = False,
            ),
            "cudnn": attr.string(
                mandatory = False,
            ),
            "tensorrt": attr.string(
                mandatory = False,
            ),
            "deepstream": attr.string(
                mandatory = False,
            ),
            "triton": attr.string(
                mandatory = False,
            ),
            "vpi": attr.string(
                mandatory = False,
            ),
            "force": attr.bool(
                default=False,
                doc="Force publish the interface if not present, this will remove all other variants",
                mandatory=False
            ),
        },
)

# This macro is used to publish an extension from local default repository to
# a user specified NGC registry.
def publish_extension(
    name,
    ext_name,
    ext_registration_target,
    repo_name,
    arch,
    distribution,
    os,
    cuda="",
    cudnn="",
    tensorrt="",
    deepstream="",
    triton="",
    vpi="",
    force=False):

    publish_extension_rule(
        name = name,
        ext_name = ext_name,
        ext_registration_target = ext_registration_target,
        repo_name = repo_name,
        arch = arch,
        distribution = distribution,
        os = os,
        cuda = cuda,
        cudnn = cudnn,
        tensorrt = tensorrt,
        deepstream = deepstream,
        triton = triton,
        vpi = vpi,
        force = force,
    )
