# Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

_SETUP_PY_TEMPLATE = """
from setuptools import setup

def readme():
    with open("{file_desc}") as f:
        return f.read()

setup(
    name = "{name}",
    version = "{version}",
    description = "{description}",
    long_description = readme(),
    classifiers=[{classifiers}],
    keywords = "{keywords}",
    url = "{url}",
    author = "{author}",
    author_email = "{author_email}",
    license = "{license}",
    packages=[{packages}],
    install_requires=[{install_requires}],
    package_data={data},
    entry_points = {etr_pt},
    zip_safe=False,
)
"""


def make_dict_bindings(bp):
    dict_bindings = {}
    for path in bp:
        file = path.split("/")[-1]
        dir_path = path[:-(len(file) + 1)]
        if dir_path not in dict_bindings:
            dict_bindings[dir_path] = []
        dict_bindings[dir_path].append(file)
    return dict_bindings


def make_entry_point(pmf):
    str_main = "registry_cli = %s" % pmf
    dict_etr = {"console_scripts": [str_main]}
    return dict_etr


def registry_package(name, version, description, file_description, classifiers, keywords, url,
                 author, author_email, license, packages, path_main_file, install_requires,
                 deps, data, visibility=["//visibility:public"]):
    data.append(file_description)
    dict_data = make_dict_bindings(data)
    dict_etr_pt = make_entry_point(path_main_file)
    setup_py = _SETUP_PY_TEMPLATE.format(
        name=name,
        version=version,
        description=description,
        file_desc=file_description,
        path_main_file=path_main_file,
        classifiers=', '.join(['"%s"' % c for c in classifiers]),
        keywords=keywords,
        url=url,
        author=author,
        author_email=author_email,
        install_requires=', '.join(['"%s"' % p for p in install_requires]),
        packages=', '.join(['"%s"' % p for p in packages]),
        data=dict_data,
        etr_pt=dict_etr_pt,
        license=license,
    )
    cmd_gen_setup = "echo '%s' > $(location setup.py)" % setup_py

    native.genrule(
        name=name + "_gen_setup",
        srcs=deps,
        outs=["setup.py"],
        cmd=cmd_gen_setup,
        visibility=visibility,
    )
    package_name = "%s-%s-py3-none-any.whl" % (name, version)

    cmd_gen_wheel = "cp -r $(GENDIR)/registry/* registry/ &&" + \
                    "mv registry/README.public %s &&" % file_description + \
                    "touch registry/bindings/__init__.py &&" + \
                    "python3  registry/setup.py bdist_wheel &&" +\
                    "cp dist/%s $(location %s) &&" % (package_name, package_name) +\
                    "echo Pip package generated at: $(location %s)" % package_name

    native.genrule(
        name=name + "_gen_pip_package",
        srcs=["setup.py"] + deps,
        outs=[package_name],
        cmd=cmd_gen_wheel,
        visibility=visibility,
    )

    bin_name = "gxf_registry"
    cmd_gen_standalone_bin = "cp -r $(GENDIR)/registry/* registry/ &&" + \
                             "mv registry/README.public %s &&" % file_description + \
                             "touch registry/bindings/__init__.py &&" + \
                             "pyinstaller --hidden-import pkg_resources.extern --onefile registry/cli/registry_cli.py && " +\
                             "cp dist/%s $(location %s) &&" % (bin_name, bin_name) +\
                             "echo Standalone binary generated at: $(location %s)" % bin_name

    native.genrule(
        name=name + "_gen_standalone_bin",
        srcs=deps,
        outs=[bin_name],
        cmd=cmd_gen_standalone_bin,
        visibility=visibility,
    )
