"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# Implementation for the expand_template rule.
def expand_template_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
        is_executable = False,
    )

# A custom bazel rule to make template substitutions to a file.
# To use this rule, you need to specify 4 fields:
# name: the name of the dependency which can be included by other bazel build rules
# template: the name of the file which will be input to this rule
# out: the name of the file which will be output by this rule
# substitutions: a dictionary mapping input tokens (read from the template file) to output tokens
#                (as they will be seen in the output file after substitution)
expand_template = rule(
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
        "out": attr.output(mandatory = True),
    },
    output_to_genfiles = True,
    implementation = expand_template_impl,
)
