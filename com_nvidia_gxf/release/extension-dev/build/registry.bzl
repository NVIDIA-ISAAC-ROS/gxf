"""
 SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FileCollector = provider(
    fields = {"manifest": "extension manifest file",
              "metadata": "extension metadata file"},
    )

def _ngc_deps_to_string(ctx):
  if not ctx.attr.ngc_dependencies:
    return "[]"

  str = ""
  for k,v in ctx.attr.ngc_dependencies.items():
    d_str  =  "\n"
    d_str +=  "- extension: " + k + "\n"
    d_str +=  "  version: " + v
    str += d_str

  return str

def _list_to_string(list_ip):
  if len(list_ip) == 0:
    return "[]"

  str = "["
  for l in list_ip:
    str += l + ", "
  str = str[:-2] + "]"
  return str

def _get_file_paths(f_list):
  files = []
  for f in f_list:
    files.append(f.path)
  return files

def _get_local_deps_paths(ctx):
  deps = []
  for d in ctx.attr.local_dependencies:
    deps.append(d[FileCollector].metadata.path)
  return deps

def _get_local_deps_files(ctx):
  deps = []
  for d in ctx.attr.local_dependencies:
    deps.append(d[FileCollector].metadata)
  return deps

def _get_python_binding_paths(ctx):
  deps = []
  for d in ctx.attr.python_bindings:
    deps.append(d.files.to_list()[0].path)
  return deps

def _get_python_src_paths(ctx):
  deps = []
  for d in ctx.attr.python_sources:
    deps.append(d.files.to_list()[0].path)
  return deps

def _verify_platform_config(ctx):
  if "arch" not in ctx.attr.platform:
    fail("platform attribute missing mandatory field \"arch\"")
  if "os" not in ctx.attr.platform:
    fail("platform attribute missing mandatory field \"os\"")
  if "distribution" not in ctx.attr.platform:
    fail("platform attribute missing mandatory field \"distribution\"")
  if ctx.attr.platform["arch"] not in ["x86_64", "aarch64", "aarch64_sbsa"]:
    fail("platform config does not match the requirement. "
    + "\"arch\" has to be one of: \"x86_64\", \"aarch64\", \"aarch64_sbsa\"")
  if ctx.attr.platform["os"] not in ["linux", "qnx"]:
    fail("platform config does not match the requirement. "
    + "\"os\" has to be one of: \"linux\"")
  if ctx.attr.platform["distribution"] not in ["ubuntu_20.04", "qnx_sdp_7.1"]:
    fail("platform config does not match the requirement. "
    + "\"distribution\" has to be one of: \"ubuntu_20.04\", \"qnx_sdp_7.1\"")

def _get_substitutions(ctx):
  subs = {"{NAME}": ctx.label.name[9:len(ctx.label.name)-4],
          "{EXTENSION_LIBRARY}": ctx.attr.extension.files.to_list()[0].path,
          "{VERSION}": ctx.attr.version,
          "{LICENSE_FILE}": ctx.file.license_file.path,
          "{UUID}": ctx.attr.uuid,
          "{URL}": ctx.attr.url,
          "{GIT_REPOSITORY}": ctx.attr.git_repository,
          "{LABELS}": _list_to_string(ctx.attr.labels),
          "{PRIORITY}": ctx.attr.priority,
          "{ARCH}": ctx.attr.platform.get("arch"),
          "{OS}": ctx.attr.platform.get("os"),
          "{DISTRIBUTION}": ctx.attr.platform.get("distribution"),
          "{CUDA}": ctx.attr.cuda,
          "{CUDNN}": ctx.attr.cudnn,
          "{TENSORRT}": ctx.attr.tensorrt,
          "{DEEPSTREAM}": ctx.attr.deepstream,
          "{TRITON}": ctx.attr.triton,
          "{VPI}": ctx.attr.vpi,
          "{HEADERS}": _list_to_string(_get_file_paths(ctx.files.headers)),
          "{BINARIES}": _list_to_string(_get_file_paths(ctx.files.binaries)),
          "{DATA}": _list_to_string(_get_file_paths(ctx.files.data)),
          "{PYTHON_ALIAS}": ctx.attr.python_alias,
          "{PYTHON_BINDINGS}": _list_to_string(_get_python_binding_paths(ctx)),
          "{PYTHON_SOURCES}": _list_to_string(_get_python_src_paths(ctx)),
          "{DEPENDENCIES}": _ngc_deps_to_string(ctx),
          }

  return subs

def _register_ext_impl(ctx):
  # Check if registration is supposed to be skipped
  if ctx.attr.skip:
    metadata_output = ctx.actions.declare_file(ctx.label.name + "_metadata.yaml")
    output_manifest = ctx.actions.declare_file(ctx.label.name + "_manifest.yaml")
    ctx.actions.write(metadata_output, content = "")
    ctx.actions.write(output_manifest, content = "")

    return [FileCollector(manifest = output_manifest, metadata = metadata_output)]

  # Verify platform config matches the requirement
  _verify_platform_config(ctx)

  # Generate intermediate ext manifest
  inter_manifest = ctx.actions.declare_file(ctx.label.name + "_inter_manifest.yaml")
  ctx.actions.expand_template(
    output = inter_manifest,
    template = ctx.file._template,
    substitutions = _get_substitutions(ctx)
  )

  if ctx.attr.platform["arch"] not in ["x86_64"]:
    output_manifest = inter_manifest

  if ctx.attr.platform["arch"] in ["x86_64"]:
    # Update dependency info in manifest
    output_manifest = ctx.actions.declare_file(ctx.label.name + "_manifest.yaml")
    args = []
    args.append("--deps")
    args.extend(_get_local_deps_paths(ctx))
    args.append("--input_manifest")
    args.append(inter_manifest.path)
    args.append("--output_manifest")
    args.append(output_manifest.path)
    action_inputs = [inter_manifest]
    action_inputs.extend(_get_local_deps_files(ctx))

    ctx.actions.run(
      outputs = [output_manifest],
      inputs = action_inputs,
      arguments = args,
      progress_message = "Fetching dependencies for %s" % ctx.attr.extension.files.to_list()[0].path,
      executable = ctx.executable._depedency_manager_tool,
      # FIXME should be able to access meta files from dep ext without disabling sandbox
      execution_requirements = {"no-sandbox": "1"},
    )

  # Register ext
  metadata_output = ctx.actions.declare_file(ctx.label.name + "_metadata.yaml")
  args = ["extn", "add", "-m", output_manifest.path, "-meta", metadata_output.path]
  action_inputs = ctx.attr.extension.files.to_list()
  action_inputs.append(output_manifest)
  if ctx.attr.local_dependencies:
      action_inputs += ctx.files.local_dependencies

  for python_binding in ctx.attr.python_bindings:
      action_inputs += python_binding.files.to_list()

  for python_source in ctx.attr.python_sources:
      action_inputs += python_source.files.to_list()

  for hdr in ctx.attr.headers:
      action_inputs += hdr.files.to_list()

  for data in ctx.attr.data:
      action_inputs += data.files.to_list()

  for binary in ctx.attr.binaries:
      action_inputs += binary.files.to_list()

  ctx.actions.run(
      outputs = [metadata_output],
      inputs = action_inputs,
      arguments = args,
      progress_message = "Registering Extension %s" % ctx.attr.extension.files.to_list()[0].path,
      executable = "registry",
      execution_requirements = {"no-sandbox": "1"},
      # use_default_shell_env = True,
  )

  return [FileCollector(manifest = output_manifest, metadata = metadata_output)]


register_ext_rule = rule(
  implementation = _register_ext_impl,
  attrs =
  {
    "_template": attr.label(
            allow_single_file = [".tpl"],
            default = "@com_extension_dev//build:registry_manifest.tpl"),
    "_depedency_manager_tool": attr.label(
        executable = True,
        cfg = "exec",
        allow_files = True,
        default = Label("//build:dependency_manager")),
    "extension": attr.label(
            allow_single_file = [".so"],
            mandatory = True),
    "version": attr.string(
            default = "1.0.0",
            doc = "version of the extension",
            mandatory = True),
    "license_file": attr.label(
            allow_single_file = ["LICENSE"],
            mandatory = True),
    "uuid": attr.string(
            doc = "uuid of the extension",
            mandatory = True),
    "url": attr.string(
            doc = "URL to extension webpage"),
    "git_repository": attr.string(
            doc = "URL to extension git repository"),
    "labels": attr.string_list(
            doc = "list of labels"),
    "priority": attr.string(
            doc = "deployment priority of extension",
            default = "P0"),
    "platform": attr.string_dict(
            doc = "deployment plaform specs",
            mandatory = True),
    "cuda": attr.string(
            doc = "cuda compute stack version"),
    "cudnn": attr.string(
            doc = "cudnn compute stack version"),
    "tensorrt": attr.string(
            doc = "tensorrt compute stack version"),
    "deepstream": attr.string(
            doc = "deepstream compute stack version"),
    "triton": attr.string(
            doc = "triton compute stack version"),
    "vpi": attr.string(
            doc = "vpi compute stack version"),
    "ngc_dependencies" : attr.string_dict(
            doc="dict containing the pairs of extensions and their corresponding"
                 + " version from NGC repositories",
            allow_empty = True,),
    "local_dependencies" : attr.label_list(
           doc="registration targets of dependent extensions from local workspace",
           allow_empty = True,),
    "metadata": attr.output(
        doc = "extension metadata output",
        mandatory = True,),
    "headers": attr.label_list(
            doc = "extension headers",
            allow_files = [".h", ".hpp", ".cuh", "BUILD.public"]),
    "binaries": attr.label_list(
            doc = "extension binaries",
            allow_files = True),
    "data": attr.label_list(
            doc = "extension data files",
            allow_files = True),
    "python_alias": attr.string(
            doc = "python alias for python bindings"),
    "python_bindings": attr.label_list(
            doc = "python bindings",
            allow_empty = True,
            allow_files = [".so"]),
    "python_sources": attr.label_list(
            doc = "python src",
            allow_empty = True,
            allow_files = [".py"]),
    "skip": attr.bool(
            doc = "Choose to skip registration",
            default = True),
  }
)

def register_extension(
        name,
        extension,
        uuid,
        version,
        license_file,
        url,
        priority,
        license = None,
        platform_config = {},
        compute_dependencies = [],
        git_repository="",
        labels = [],
        badges = [],
        ngc_dependencies = {},
        local_dependencies = [],
        headers = [],
        binaries = [],
        data = [],
        python_bindings = [],
        python_sources = [],
        python_alias = "",
        visibility = None):
  """
    A macro wrapping register_ext_rule which updates the extension lib
    target name using the same convention as gxf core

  """

  # Use platform default config if it's not explicitly set
  if not platform_config:
    X86_LINUX_UB_20 = { "arch" : "x86_64", "os": "linux", "distribution" : "ubuntu_20.04"}
    platform_config = select({
        "//conditions:default": X86_LINUX_UB_20,
        "@com_extension_dev//build:platform_x86_64": X86_LINUX_UB_20,
        "@com_extension_dev//build:cpu_aarch64": {
            "arch" : "aarch64",
            "os": "linux",
            "distribution" : "ubuntu_20.04",
        },
    })

  # Check for invalid compute dependencies
  for dependency in compute_dependencies:
    if dependency not in ["cuda", "cudnn", "tensorrt", "deepstream", "triton", "vpi"]:
      fail("Unsupported dependency :" + dependency)

  # Select compute stack versions based on the configs used to build the extension
  if "cuda" in compute_dependencies:
    cuda = select({
      "//conditions:default": "11.7",
      "@com_extension_dev//build:platform_x86_64": "11.8",
      "@com_extension_dev//build:cpu_aarch64": "11.4",
    })
  else:
    cuda = None

  if "cudnn" in compute_dependencies:
    cudnn = select({
      "//conditions:default": "8.4.1",
      "@com_extension_dev//build:platform_x86_64": "8.6.0",
      "@com_extension_dev//build:cpu_aarch64": "8.6.0",
    })
  else:
    cudnn = None

  if "tensorrt" in compute_dependencies:
    tensorrt = select({
      "//conditions:default": None,
      "@com_extension_dev//build:platform_x86_64": "8.5.1",
      "@com_extension_dev//build:cpu_aarch64": "8.5.1",
    })
  else:
    tensorrt = None

  if "deepstream" in compute_dependencies:
    deepstream = select({
      "//conditions:default": None,
      "@com_extension_dev//build:platform_x86_64": "6.2.1",
      "@com_extension_dev//build:cpu_aarch64": "6.2.1",
      })
  else:
    deepstream = None

  if "triton" in compute_dependencies:
    triton = select({
      "//conditions:default": None,
      "@com_extension_dev//build:platform_x86_64": "2.26.0",
      "@com_extension_dev//build:cpu_aarch64": "2.29.0",
      })
  else:
    triton = None

  if "vpi" in compute_dependencies:
    vpi = select({
      "//conditions:default": None,
      "@com_extension_dev//build:platform_x86_64": "2.1.6",
      "@com_extension_dev//build:cpu_aarch64": "2.1.6",
      })
  else:
    vpi = None

  # clean up target label
  # Example: //gxf/test/extensions:test to //gxf/test/extensions:libgxf_test.so
  ext_split = extension.split(":")
  if len(ext_split) > 1:
    extension_library = ext_split[0] + ":libgxf_" + ext_split[-1] + ".so"
  else:
    extension_library = "libgxf_" + extension + ".so"

  if not python_alias:
    python_alias = extension

  register_ext_rule(
    name = name,
    extension = extension_library,
    uuid = uuid,
    version = version,
    license_file = license_file,
    url = url,
    git_repository = git_repository,
    labels = labels,
    priority = priority,
    platform = platform_config,
    cuda = cuda,
    cudnn = cudnn,
    tensorrt = tensorrt,
    deepstream = deepstream,
    triton = triton,
    vpi = vpi,
    ngc_dependencies = ngc_dependencies,
    local_dependencies = local_dependencies,
    visibility = visibility,
    headers = headers,
    binaries = binaries,
    data = data,
    python_alias = python_alias,
    python_bindings = python_bindings,
    python_sources = python_sources,
    metadata = name + "_metadata.yaml",
    skip = select({
                   "@com_extension_dev//build:skip_registration": True,
                   "//conditions:default": False,
                 }),
  )
