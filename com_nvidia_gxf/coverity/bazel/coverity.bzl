"""
Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load(
    "@compdb//:aspects.bzl",
    "compilation_database_aspect",
    "CompilationAspect",
)

_coverity_tag = "coverity"
_coverity_config = "@com_nvidia_gxf//coverity/config"

_cov_dir = "cov_dir"

_coverity_script_header = """\
#!/bin/bash

pwd

export PATH={cov_tools_path}:$PATH
echo $PATH

which cov-translate || echo Failed to locate \'cov-translate\' 1>&2 || exit -1
which cov-analyze || echo Failed to locate \'cov-analyze\' 1>&2 || exit -1
which cov-format-errors || echo Failed to locate \'cov-format-errors\' 1>&2 || exit -1

"""

_cov_translate_template = """\
cmd="cov-translate \
--dir {cov_database_dir} \
--config {cov_config} \
{options} \
{compile_cmd} \
"
eval $cmd || echo \'cov-translate\' failure || exit -1
echo
"""

_coverity_script_finish = """\
cmd="cov-analyze \
--security \
--enable INTEGER_OVERFLOW \
--enable AUDIT.SPECULATIVE_EXECUTION_DATA_LEAK \
--enable COM.BSTR.ALLOC \
--config {cov_config} \
--jobs 1 \
--strip-path `pwd` \
--ticker-mode none \
--dir {cov_database_dir} \
{coding_standard} \
{options} \
"
eval $cmd || echo \'cov-analyze\' failure || exit -1
echo

cmd="cov-format-errors \
--config {cov_config} \
--dir {cov_database_dir} \
--json-output-v7 {json_report_file}
"

eval $cmd || echo \'cov-format-errors\' failure || exit -1
echo

if [[ -s {json_report_file} ]]; then
    cmd="cov-format-errors \
    --config {cov_config} \
    --dir {cov_database_dir} \
    --html-output {html_report_dir} \
    --include-files '{include_files}' \
    --strip-path `pwd` \
    -x
    "
    echo $cmd
    eval $cmd || echo Failed to generate HTML report || exit -1

    if  [ "$(cat {html_report_dir}/index.html | md5sum)" = "a2131aaaf7de1d7ce59aadf33491d59f  -" ]
    then
        echo "No violations detected for target \'{target_name}\'."
        rm -rf {html_report_dir}
        exit 0
    else
        echo -e "Coverity violation detected for target \'{target_name}\'."
        echo -e "JSON report available at $(realpath {json_report_file})"
        echo -e "HTML report available at $(realpath {html_report_dir})/index.html"
        exit -1
    fi
fi
"""

def _has_code(srcs, hdrs = []):
    return len(srcs + hdrs) > 0

def coverity(name, tags = []):
    """
    Coverity report generation as a build rule. Fails in case of any violation.
    """
    coverity_test(
        name = "_coverity_" + name,
        src = name,
        tags = tags + [_coverity_tag, "no-sandbox"],
        config = _coverity_config,
        translate_options = [
            "--verbose 1",
            "--emit-complementary-info",
        ],
        coverity_tools = "@coverity//:tools",
        run_coverity = select({
            "//conditions:default": False,
            "@com_nvidia_gxf//coverity:coverity_build": True,
        }),
        coding_standard_config = select({
            "//conditions:default": None,
            "@com_nvidia_gxf//coverity:none": None,
            "@com_nvidia_gxf//coverity:autosar-only": "@com_nvidia_gxf//coverity/ruleset:autosarcpp14-required-only.config",
        })
    )

def _full_target_name(label):
    '''
    Assembles full target name in form of `@repo//full/package/path:target` from `Label` object.
    '''
    if type(label) != "Label":
        fail("type of `label` must be `Label`")

    full_name = "//{package}:{name}".format(
        repo = label.workspace_name,
        package = label.package,
        name = label.name,
    )

    if len(label.workspace_name):
        full_name = "@{repo}" + full_name

    return full_name

def _coverity_test_impl(ctx):
    if ctx.attr.run_coverity == False:
        test_script = ctx.actions.declare_file(ctx.attr.name + ".sh")
        ctx.actions.write(output = test_script, content = "#noop", is_executable = True)

        return DefaultInfo(executable = test_script)

    compdb = ctx.attr.src[CompilationAspect].compilation_db
    src_files = ctx.attr.src[OutputGroupInfo].source_files.to_list()
    hdr_files = ctx.attr.src[OutputGroupInfo].header_files.to_list()

    if len(src_files) == 0:
        if ctx.attr.mandatory:
            fail("`src` must be a target with at least one source or header.")
        else:
            test_script = ctx.actions.declare_file(ctx.attr.name + ".sh")
            ctx.actions.write(output = test_script, content = "#noop", is_executable = True)

            return DefaultInfo(executable = test_script)

    cov_tools_files = ctx.attr.coverity_tools.files.to_list()
    cov_tools_path = cov_tools_files[0].dirname

    standard_files = ctx.attr.coding_standard_config.files.to_list() if ctx.attr.coding_standard_config else []
    standard_path = " ".join(["--coding-standard-config " + standard_file.path for standard_file in standard_files]) if standard_files else ""
    conf_files = ctx.attr.config.files.to_list()
    cov_config = conf_files[0]

    runfiles = src_files + hdr_files + cov_tools_files + standard_files + conf_files

    src_file_list = [ f.path for f in src_files ]

    translate_options = " ".join(ctx.attr.translate_options)
    content = _coverity_script_header.format(cov_tools_path = cov_tools_path)
    for entry in compdb.to_list():
        if entry.file in src_file_list:
            cli = entry.command.replace(
                "external/toolchain/crosstool/scripts/crosstool_wrapper_driver_is_not_gcc_host.py",
                "/usr/bin/gcc",
            )
            content += _cov_translate_template.format(
                cov_database_dir = _cov_dir,
                cov_config = cov_config.path,
                options = translate_options,
                compile_cmd = cli,
            )

    src_target_name = _full_target_name(ctx.attr.src.label)
    analyze_options = " ".join(ctx.attr.analyze_options)
    files_regexp = "(" + "|".join([f.path for f in src_files]) + ")$"

    content += _coverity_script_finish.format(
        cov_config = cov_config.path,
        cov_database_dir = _cov_dir,
        json_report_file = ctx.attr.name + ".json",
        html_report_dir = ctx.attr.name + "_html_report",
        include_files = files_regexp,
        options = analyze_options,
        coding_standard = standard_path,
        target_name = src_target_name,
    )

    test_script = ctx.actions.declare_file(ctx.outputs.executable.basename)
    ctx.actions.write(output = test_script, content = content, is_executable = True)

    return DefaultInfo(
        files = depset([test_script]), runfiles = ctx.runfiles(files=runfiles)
    )

coverity_test = rule(
    attrs = {
        "src": attr.label(
            aspects = [compilation_database_aspect],
            doc = "Source target to run coverity on.",
            mandatory = True,
        ),
        "mandatory": attr.bool(
            default = False,
            doc = "Throw error if `src` is not eligible for linter check, e.g. have no C/C++ source or header.",
        ),
        "config": attr.label(
            doc = "Coverity configuration file",
            mandatory = True,
        ),
        "coding_standard_config": attr.label(
            doc = "Coverity Coding Standard Config",
            allow_single_file = True,
        ),
        "translate_options": attr.string_list(
            doc = "CLI options for `cov-translate`",
            default = ["--no-headers"],
        ),
        "analyze_options": attr.string_list(
            doc = "CLI options for `cov-analyze`",
            default = [],
        ),
        "coverity_tools": attr.label(
            doc = "Coverity tools",
            mandatory = True,
        ),
        "run_coverity": attr.bool(
            doc = "Flag to check whether to run coverity",
            default = False,
        ),
    },
    implementation = _coverity_test_impl,
    test = True,
)
