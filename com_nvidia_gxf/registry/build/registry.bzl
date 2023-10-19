"""
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")

load("@com_nvidia_gxf//registry/build:defs.bzl",
     "get_platform_arch",
     "get_platform_os",
     "get_platform_os_distribution",
     "get_cuda_version",
     "get_cudnn_version",
     "get_tensorrt_version",
     "get_deepstream_version",
     "get_triton_version",
     "get_vpi_version")

# A bazel provider to return all metadata info and runfiles
# from a gxf extension registration target
FileCollector = provider(
    fields = {"manifest": "extension manifest file",
              "metadata": "extension metadata file",
              "base_manifests": "list of extension base manifest files",
              "runfiles": "Runfiles needed for registration"},
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

def _list_to_string_bash(list_ip):
  if len(list_ip) == 0:
    return "()"

  str = "("
  for l in list_ip:
    str += '"' + l + '" '
  str = str[:-1] + ')'

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

def _get_binaries(binaries, arch):
  bin = []
  for d in binaries:
    if arch == "x86_64":
      bin.append(d.path)
    else:
      bin.append(d.short_path)

  return bin

def _get_local_deps_runfiles(ctx):
  deps = []
  for d in ctx.attr.local_dependencies:
    deps.extend(d[FileCollector].runfiles)
  return deps

def _get_local_deps_short_paths(ctx):
  deps = []
  for d in ctx.attr.local_dependencies:
    # Load base extensions of a dep before loading the dep itself
    for e in d[FileCollector].base_manifests:
      if e.short_path not in deps:
        deps.append(e.short_path)
    if d[FileCollector].manifest.short_path not in deps:
      deps.append(d[FileCollector].manifest.short_path)

  return deps

def _get_local_deps_paths(ctx):
  deps = []
  for d in ctx.attr.local_dependencies:
    deps.append(d[FileCollector].manifest.path)
  return deps

def _get_local_deps_files(ctx):
  deps = []
  for d in ctx.attr.local_dependencies:
    # Load base extensions of a dep before loading the dep itself
    for e in d[FileCollector].base_manifests:
      if e not in deps:
        deps.append(e)
    if d[FileCollector].manifest not in deps:
      deps.append(d[FileCollector].manifest)

  return deps

def _get_python_binding_paths(ctx):
  deps = []
  for d in ctx.attr.python_bindings:
    deps.append(d.files.to_list()[0].short_path)
  return deps

def _get_python_src_paths(ctx):
  deps = []
  for d in ctx.attr.python_sources:
    if ctx.attr.arch == "x86_64":
      deps.append(d.files.to_list()[0].path)
    else:
      deps.append(d.files.to_list()[0].short_path)

  return deps

def _get_extension_lib_path(ctx):
  if ctx.attr.arch == "x86_64":
    return ctx.attr.extension.files.to_list()[0].path
  return ctx.attr.extension.files.to_list()[0].short_path

def _verify_platform_config(ctx):
  if ctx.attr.arch not in ["x86_64", "aarch64", "aarch64_sbsa"]:
    fail("platform arch config does not match the requirement. "
    + "\"arch\" has to be one of: \"x86_64\", \"aarch64\", \"aarch64_sbsa\"")
  if ctx.attr.os not in ["linux", "qnx"]:
    fail("platform os config does not match the requirement. "
    + "\"os\" has to be one of: \"linux\"")
  if ctx.attr.distribution not in ["ubuntu_20.04", "qnx_sdp_7.1"]:
    fail("platform distribution config does not match the requirement. "
    + "\"distribution\" has to be one of: \"ubuntu_20.04\", \"qnx_sdp_7.1\"")

def _get_substitutions(ctx):
  # We require that the label name is "register_foo_ext":
  if not ctx.label.name.startswith("register_"):
    fail("Expected name of target to start with prefix \"register_\"."
    + " Current name is \"{}\".".format(ctx.label.name))

  if not ctx.label.name.endswith("_ext"):
    fail("Expected name of target to end with suffix \"_ext\"."
    + "Current name is \"{}\".".format(ctx.label.name))

  subs = {"{NAME}": ctx.label.name[9:len(ctx.label.name)-4],
          "{EXTENSION_LIBRARY}": _get_extension_lib_path(ctx),
          "{VERSION}": ctx.attr.version,
          "{LICENSE_FILE}": ctx.file.license_file.path,
          "{UUID}": ctx.attr.uuid,
          "{URL}": ctx.attr.url,
          "{GIT_REPOSITORY}": ctx.attr.git_repository,
          "{LABELS}": _list_to_string(ctx.attr.labels),
          "{PRIORITY}": ctx.attr.priority,
          "{ARCH}": ctx.attr.arch,
          "{OS}": ctx.attr.os,
          "{DISTRIBUTION}": ctx.attr.distribution,
          "{CUDA}": ctx.attr.cuda,
          "{CUDNN}": ctx.attr.cudnn,
          "{TENSORRT}": ctx.attr.tensorrt,
          "{DEEPSTREAM}": ctx.attr.deepstream,
          "{TRITON}": ctx.attr.triton,
          "{VPI}": ctx.attr.vpi,
          "{HEADERS}": _list_to_string(_get_file_paths(ctx.files.headers)),
          "{BINARIES}": _list_to_string(_get_binaries(ctx.files.binaries, ctx.attr.arch)),
          "{DATA}": _list_to_string(_get_file_paths(ctx.files.data)),
          "{PYTHON_ALIAS}": ctx.attr.python_alias,
          "{NAMESPACE}": ctx.attr.namespace,
          "{PYTHON_BINDINGS}": _list_to_string(_get_python_binding_paths(ctx)),
          "{PYTHON_SOURCES}": _list_to_string(_get_python_src_paths(ctx)),
          "{DEPENDENCIES}": _ngc_deps_to_string(ctx),
          }

  return subs

def _register_ext_impl(ctx):
  # Verify platform config matches the requirement
  _verify_platform_config(ctx)

  # Generate intermediate ext manifest
  inter_manifest = ctx.actions.declare_file(ctx.label.name + "_inter_manifest.yaml")
  ctx.actions.expand_template(
    output = inter_manifest,
    template = ctx.file._template,
    substitutions = _get_substitutions(ctx)
  )

  if ctx.attr.arch not in ["x86_64"]:
    output_manifest = inter_manifest

  # Register extension. Collect all registration inputs
  action_inputs = ctx.attr.extension.files.to_list()
  action_inputs += ctx.attr._registry[DefaultInfo].data_runfiles.files.to_list()
  action_inputs += ctx.attr._registry[DefaultInfo].default_runfiles.files.to_list()

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

  # Uncomment below line to make license file mandatory for all extensions
  # action_inputs.append(ctx.file.license_file)

  # Add all files from dependent extensions
  action_inputs += _get_local_deps_files(ctx) + _get_local_deps_runfiles(ctx)

  if ctx.attr.arch in ["x86_64"]:
    # Update dependency info in manifest
    output_manifest = ctx.actions.declare_file(ctx.label.name + "_manifest.yaml")
    args = []
    args.append("--deps")
    args.extend(_get_local_deps_paths(ctx))
    args.append("--input_manifest")
    args.append(inter_manifest.path)
    args.append("--output_manifest")
    args.append(output_manifest.path)
    action_inputs.append(inter_manifest)

    ctx.actions.run(
      outputs = [output_manifest],
      inputs = action_inputs,
      arguments = args,
      progress_message = "Fetching dependencies for %s" % ctx.attr.extension.files.to_list()[0].path,
      executable = ctx.executable._depedency_manager_tool,
      # FIXME should be able to access meta files from dep ext without disabling sandbox
      execution_requirements = {"no-sandbox": "1"},
    )

  action_inputs.append(output_manifest)

  # Registration output metadata file
  metadata_output = ctx.label.name + "_metadata.yaml"

  # generate registration script
  register_script = ctx.actions.declare_file(ctx.label.name + ".sh")
  ctx.actions.expand_template(
      template = ctx.file._register_template,
      output = register_script,
      substitutions = {
          "{EXTENSION_MANIFEST}": output_manifest.short_path,
          "{EXTENSION_METADATA}": metadata_output,
          "{DEPENDENCIES}": _list_to_string_bash(_get_local_deps_short_paths(ctx)),
      },
      is_executable = True,
  )

  return [DefaultInfo(executable = register_script,
                      runfiles=ctx.runfiles(action_inputs)),
          FileCollector(manifest = output_manifest,
                        metadata = metadata_output,
                        base_manifests = _get_local_deps_files(ctx),
                        runfiles=action_inputs)]


register_ext_rule = rule(
  implementation = _register_ext_impl,
  executable = True,
  attrs =
  {
    "_template": attr.label(
            allow_single_file = [".tpl"],
            default = "//registry/build:registry_manifest.tpl"),
    "_register_template": attr.label(
            allow_single_file = [".sh.tpl"],
            default = "//registry/build:register.sh.tpl"),
    "_depedency_manager_tool": attr.label(
        executable = True,
        cfg = "exec",
        allow_files = True,
        default = Label("//registry/core:dependency_manager")),
    "_registry": attr.label(
            doc = "registry cli tool used to install graphs before running them",
            default = "//registry/cli:registry_cli"),
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
    "arch": attr.string(
            doc = "deployment plaform arch",
            mandatory = True),
    "os": attr.string(
            doc = "deployment plaform os",
            mandatory = True),
    "distribution": attr.string(
            doc = "deployment plaform distribution",
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
    "metadata": attr.string(
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
    "namespace": attr.string(
            doc = "namespace of the extension. e.g gxf, isaac, etc"),
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
        arch = None,
        os = None,
        distribution = None,
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
        namespace='gxf',
        visibility = None):
  """
    A macro wrapping register_ext_rule which updates the extension lib
    target name using the same convention as gxf core

  """

  # Use platform default config if it's not explicitly set
  if not arch:
    arch = get_platform_arch()

  if not os:
    os = get_platform_os()

  if not distribution:
    distribution = get_platform_os_distribution()

  # Check for invalid compute dependencies
  for dependency in compute_dependencies:
    if dependency not in ["cuda", "cudnn", "tensorrt", "deepstream", "triton", "vpi"]:
      fail("Unsupported dependency :" + dependency)

  # Select compute stack versions based on the configs used to build the extension
  if "cuda" in compute_dependencies:
    cuda = get_cuda_version()
  else:
    cuda = None

  if "cudnn" in compute_dependencies:
    cudnn = get_cudnn_version()
  else:
    cudnn = None

  if "tensorrt" in compute_dependencies:
    tensorrt = get_tensorrt_version()
  else:
    tensorrt = None

  if "deepstream" in compute_dependencies:
    deepstream = get_deepstream_version()
  else:
    deepstream = None

  if "triton" in compute_dependencies:
    triton = get_triton_version()
  else:
    triton = None

  if "vpi" in compute_dependencies:
    vpi = get_vpi_version()
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
    arch = arch,
    os = os,
    distribution = distribution,
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
    namespace = namespace,
    python_alias = python_alias,
    python_bindings = python_bindings,
    python_sources = python_sources,
    metadata = name + "_metadata.yaml",
  )
