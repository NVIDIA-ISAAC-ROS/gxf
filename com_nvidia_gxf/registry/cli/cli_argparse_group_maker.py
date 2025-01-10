# Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" The purpose of the following functions is to create the parser object
    for the Registry CLI.
"""

import argparse


def make_extension_group(subpar):
    parser_extn = subpar.add_parser("extn", help="Perform actions on "
                                                 "extensions")
    parser_extn.set_defaults(parser_name="extn")
    subpars_extn = parser_extn.add_subparsers(metavar="")

    parser_add = subpars_extn \
        .add_parser("add", help="Add extension to repository. Manifest file "
                                "includes all information about extension")
    parser_add.set_defaults(subparser_name="add", help="add extension")
    required_grp_add = parser_add.add_argument_group("required ")
    required_grp_add.add_argument("-m", "--manifest-name",
                                  dest="mnf_name",
                                  nargs=1,
                                  required=True,
                                  help="Specify the manifest name")
    parser_add.add_argument("-meta", "--metadata-file",
                            dest="meta_file",
                            required=False, nargs=1,
                            help="Specify a file name to output logs")

    parser_remove = subpars_extn \
        .add_parser("remove", help="Remove extension from repository")
    parser_remove.set_defaults(subparser_name="remove")

    parser_remove_interface = argparse.ArgumentParser(add_help=False)
    required_grp_interface = parser_remove_interface.add_argument_group("required arguments")
    required_grp_interface.add_argument("-n", "--extn-name", dest="extn_name",
                                        required=True, nargs=1, help="Select an extension name")
    required_grp_interface.add_argument("-s", "--select-version", dest="select_version",
                                        required=True, nargs=1,
                                        help="Select an extension version")
    required_grp_interface.add_argument("-r", "--repo-name", dest="repo_name",
                                        required=True, nargs=1,
                                        help="Select a repository name")

    parser_remove_variant = argparse.ArgumentParser(add_help=False)

    required_grp_variant = parser_remove_variant.add_argument_group("required arguments")
    required_grp_variant.add_argument("-n", "--extn-name", dest="extn_name",
                                      required=True, nargs=1, help="Select an extension name")
    required_grp_variant.add_argument("-s", "--select-version", dest="select_version",
                                        required=True, nargs=1,
                                        help="Select an extension version")
    required_grp_variant.add_argument("-r", "--repo-name", dest="repo_name",
                                   required=True, nargs=1,
                                   help="Select a repository name")
    required_grp_variant.add_argument("-a", "--arch",
                                    dest="arch",
                                    required=True, nargs=1,
                                    help="Select an arch from: aarch64, x86_64, aarch64_sbsa")
    required_grp_variant.add_argument("-f", "--distro",
                                    dest="distribution",
                                    required=True, nargs=1,
                                    help="Select an OS distro from: ubuntu_22.04")
    required_grp_variant.add_argument("-o", "--os", dest="os",
                                    required=True, nargs=1,
                                    help="Select an os from: Linux")
    parser_remove_variant.add_argument("--cuda", dest="cuda", nargs=1,
                                    help="Set cuda version e.g. 11.6")
    parser_remove_variant.add_argument("--cudnn",
                                    dest="cudnn",
                                    nargs=1,
                                    help="Set cudnn version e.g. 8.0.3")
    parser_remove_variant.add_argument("--tensorrt",
                                    dest="tensorrt",
                                    nargs=1,
                                    help="Set tensorrt version e.g. 8.0.1.6")
    parser_remove_variant.add_argument("--deepstream", dest="deepstream", nargs=1,
                                    help="Set deepstream version e.g. 6.0")
    parser_remove_variant.add_argument("--triton",
                                    dest="triton",
                                    nargs=1,
                                    help="Set triton version e.g. 2.51")
    parser_remove_variant.add_argument("--vpi",
                                    dest="vpi",
                                    nargs=1,
                                    help="Set vpi version e.g. 1.15.1")

    # parser_remove.add_argument("-u", "--uuid", dest="uuid",
    #                            nargs=1,
    #                            help="Select a uuid")


    parser_remove_all = argparse.ArgumentParser(add_help=False)
    required_grp_all = parser_remove_all.add_argument_group(
        "required arguments")
    required_grp_all.add_argument("-r", "--repo-name", dest="repo_name",
                                  required=True, nargs=1,
                                  help="Specify the name of the repository")

    parser_remove_one = argparse.ArgumentParser(add_help=False)
    required_grp_one = parser_remove_one.add_argument_group("required arguments")
    required_grp_one.add_argument("-n", "--extn-name", dest="extn_name",
                                      required=True, nargs=1, help="Select an extension name")
    required_grp_one.add_argument("-s", "--select-version", dest="select_version",
                                        required=True, nargs=1,
                                        help="Select a version")

    subparsers_remove = parser_remove.add_subparsers(metavar="")
    parser_remove_interface.set_defaults(subsubparser_name="interface")
    parser_remove_variant.set_defaults(subsubparser_name="variant")
    parser_remove_all.set_defaults(subsubparser_name="all")
    parser_remove_one.set_defaults(subsubparser_name="one")
    subparsers_remove.add_parser("interface", parents=[parser_remove_interface],
                                    help="Remove an extension interface")
    subparsers_remove.add_parser("variant", parents=[parser_remove_variant],
                                    help="Remove an extension variant")
    subparsers_remove.add_parser("all", parents=[parser_remove_all],
                              help="Remove all extensions in from a repository")
    subparsers_remove.add_parser("one", parents=[parser_remove_one],
                              help="Remove one extensions from a repository")

    parser_list = subpars_extn \
        .add_parser("list", help="List of extensions from all repositories")
    parser_list.set_defaults(subparser_name="list")
    parser_list.add_argument("-r", "--repo-name", dest="repo_name",
                             nargs=1, help="Select a repository name")
    parser_list.add_argument("-n", "--name", dest="name",
                             nargs=1, help="Select an extension name")
    parser_list.add_argument("-a", "--author", dest="author",
                             nargs=1, help="Filter the list by specifying "
                                           "the author")
    parser_list.add_argument("-l", "--labels", dest="labels",
                             nargs="+", help="Filter the list by specifying "
                                             "one or several labels")

    parser_publish = subpars_extn \
        .add_parser("publish", help="Publish extension to a NGC repository")
    parser_publish.set_defaults(subparser_name="publish")

    parser_publish_interface = argparse.ArgumentParser(add_help=False)
    required_grp_interface = parser_publish_interface.add_argument_group("required arguments")
    required_grp_interface.add_argument("-n", "--extn-name", dest="extn_name",
                                        required=True, nargs=1, help="Select an extension name")
    required_grp_interface.add_argument("-r", "--repo-name", dest="repo_name",
                                        required=True, nargs=1,
                                        help="Select a repository name")
    parser_publish_interface.add_argument("--log-file",
                                    dest="log",
                                    nargs=1,
                                    help="Save the extn publish logs")
    parser_publish_interface.add_argument("--force",
                                    dest="force",
                                    action="store_true",
                                    help="Overwrite the extension if its already present in NGC. This"
                                          " command will remove all variants of the current version from NGC.")

    parser_publish_variant = argparse.ArgumentParser(add_help=False)

    required_grp_variant = parser_publish_variant.add_argument_group("required arguments")
    required_grp_variant.add_argument("-n", "--extn-name", dest="extn_name",
                                      required=True, nargs=1, help="Select an extension name")
    required_grp_variant.add_argument("-r", "--repo-name", dest="repo_name",
                                   required=True, nargs=1,
                                   help="Select a repository name")
    required_grp_variant.add_argument("-a", "--arch",
                                    dest="arch",
                                    required=True, nargs=1,
                                    help="Select an arch from: aarch64, x86_64, aarch64_sbsa")
    required_grp_variant.add_argument("-f", "--distro",
                                    dest="distribution",
                                    required=True, nargs=1,
                                    help="Select an OS distro from: ubuntu_22.04")
    required_grp_variant.add_argument("-o", "--os", dest="os",
                                    required=True, nargs=1,
                                    help="Select an OS from: Linux")
    parser_publish_variant.add_argument("--cuda", dest="cuda", nargs=1,
                                    help="Set cuda version e.g.11.6")
    parser_publish_variant.add_argument("--cudnn",
                                    dest="cudnn",
                                    nargs=1,
                                    help="Set cudnn version e.g. 8.0.3")
    parser_publish_variant.add_argument("--tensorrt",
                                    dest="tensorrt",
                                    nargs=1,
                                    help="Set tensorrt version e.g. 8.0.1.6")
    parser_publish_variant.add_argument("--deepstream", dest="deepstream", nargs=1,
                                    help="Set deepstream version e.g. 6.0")
    parser_publish_variant.add_argument("--triton",
                                    dest="triton",
                                    nargs=1,
                                    help="Set triton version e.g. 2.51")
    parser_publish_variant.add_argument("--vpi",
                                    dest="vpi",
                                    nargs=1,
                                    help="Set vpi version e.g. 1.15.1")
    parser_publish_variant.add_argument("--log-file",
                                    dest="log",
                                    nargs=1,
                                    help="Save the extn publish logs")
    # parser_publish.add_argument("-u", "--uuid", dest="uuid",
    #                            nargs=1,
    #                            help="Select a uuid")

    parser_publish_all = argparse.ArgumentParser(add_help=False)
    required_grp_all = parser_publish_all.add_argument_group(
        "required arguments")
    required_grp_all.add_argument("-r", "--repo-name", dest="repo_name",
                                  required=True, nargs=1,
                                  help="Specify the name of the repository")
    parser_publish_all.add_argument("--force",
                                    dest="force",
                                    action="store_true",
                                    help="Overwrite the extension if its already present in NGC. This"
                                          " command will remove all variants of the current version from NGC.")

    parser_publish_one = argparse.ArgumentParser(add_help=False)
    required_grp_one = parser_publish_one.add_argument_group("required arguments")
    required_grp_one.add_argument("-r", "--repo-name", dest="repo_name",
                                  required=True, nargs=1,
                                  help="Specify the name of the repository")
    required_grp_one.add_argument("-n", "--extn-name", dest="extn_name",
                                      required=True, nargs=1, help="Select an extension name")
    parser_publish_one.add_argument("--force",
                                    dest="force",
                                    action="store_true",
                                    help="Overwrite the extension if its already present in NGC. This"
                                          " command will remove all variants of the current version from NGC.")


    subparsers_publish = parser_publish.add_subparsers(metavar="")
    parser_publish_interface.set_defaults(subsubparser_name="interface")
    parser_publish_variant.set_defaults(subsubparser_name="variant")
    parser_publish_all.set_defaults(subsubparser_name="all")
    parser_publish_one.set_defaults(subsubparser_name="one")

    subparsers_publish.add_parser("interface", parents=[parser_publish_interface],
                                    help="Publish an extension interface")
    subparsers_publish.add_parser("variant", parents=[parser_publish_variant],
                                    help="Publish an extension variant")
    subparsers_publish.add_parser("all", parents=[parser_publish_all],
                              help="Publish all extensions in default repository")
    subparsers_publish.add_parser("one", parents=[parser_publish_one],
                              help="Publish an extension in default repository")

    parser_info = subpars_extn \
        .add_parser("info", help="Print extension information including list"
                                 " of components")
    parser_info.set_defaults(subparser_name="info")
    required_grp_info = parser_info.add_argument_group("required arguments")
    required_grp_info.add_argument("-n", "--extn-name", dest="extn_name",
                                   required=True, nargs=1,
                                   help="Select an extension name")

    parser_import = subpars_extn \
        .add_parser("import", help="Import an extension")
    parser_import.set_defaults(subparser_name="import")

    parser_import_interface = argparse.ArgumentParser(add_help=False)
    required_grp_interface = parser_import_interface.add_argument_group("required arguments")
    required_grp_interface.add_argument("-n", "--extn-name", dest="extn_name",
                                        required=True, nargs=1, help="Select an extension name")
    required_grp_interface.add_argument("-s", "--select-version", dest="select_version",
                                        required=True, nargs=1,
                                        help="Select a version")
    required_grp_interface.add_argument("-d", "--output-directory", dest="directory",
                                       required=True, nargs=1,
                                       help="Select an output directory")

    parser_import_variant = argparse.ArgumentParser(add_help=False)
    required_grp_variant = parser_import_variant.add_argument_group("required arguments")
    required_grp_variant.add_argument("-n", "--extn-name", dest="extn_name",
                                        required=True, nargs=1,
                                        help="Select an extension name")
    required_grp_variant.add_argument("-s", "--select-version", dest="select_version",
                                        required=True, nargs=1,
                                        help="Select an extension version")
    required_grp_variant.add_argument("-a", "--arch",
                                      dest="arch",
                                      required=True, nargs=1,
                                      help="Select an arch from aarch64, x86_64, aarch64_sbsa"
                                           " Default value: x86_64")
    required_grp_variant.add_argument("-f", "--distro",
                                       dest="distribution",
                                       required=True, nargs=1,
                                       help="Select an OS distro from ubuntu_22.04"
                                            " Default value: ubuntu_22.04")
    required_grp_variant.add_argument("-o", "--os", dest="os",
                                       required=True, nargs=1,
                                       help="Select an OS from Linux"
                                            " Default value: Linux")
    required_grp_variant.add_argument("-d", "--output-directory", dest="directory",
                                       required=True, nargs=1,
                                       help="Select an output directory")
    parser_import_variant.add_argument("--cuda", dest="cuda", nargs=1,
                                    help="Set cuda version e.g. 11.6")
    parser_import_variant.add_argument("--cudnn",
                                    dest="cudnn",
                                    nargs=1,
                                    help="Set cudnn version e.g. 8.0.3")
    parser_import_variant.add_argument("--tensorrt",
                                    dest="tensorrt",
                                    nargs=1,
                                    help="Set tensorrt version e.g. 8.0.1.6")
    parser_import_variant.add_argument("--deepstream", dest="deepstream", nargs=1,
                                    help="Set deepstream version e.g. 6.0")
    parser_import_variant.add_argument("--triton",
                                    dest="triton",
                                    nargs=1,
                                    help="Set triton version e.g. 2.51")
    parser_import_variant.add_argument("--vpi",
                                    dest="vpi",
                                    nargs=1,
                                    help="Set vpi version e.g. 1.15.1")

    subparsers_import = parser_import.add_subparsers(metavar="")
    parser_import_interface.set_defaults(subsubparser_name="interface")
    parser_import_variant.set_defaults(subsubparser_name="variant")
    subparsers_import.add_parser("interface", parents=[parser_import_interface],
                                    help="Import an extension interface")
    subparsers_import.add_parser("variant", parents=[parser_import_variant],
                                    help="Import an extension variant")

    # parser_import.add_argument("-u", "--uuid", dest="uuid",
    #                              nargs=1,
    #                              help="Select a uuid")

    parser_sync = subpars_extn \
        .add_parser("sync", help="Synchronize an extension")
    parser_sync.set_defaults(subparser_name="sync")
    required_grp_sync = parser_sync.add_argument_group("required"
                                                       " arguments")
    required_grp_sync.add_argument("-r", "--repo-name", dest="repo_name",
                                   required=True, nargs=1,
                                   help="Select a repository name")
    required_grp_sync.add_argument("-e", "--extension-name", dest="extn_name",
                                   required=True, nargs=1,
                                   help="Select an extension name")
    required_grp_sync.add_argument("-s", "--select-version", dest="select_version",
                                   required=True, nargs=1,
                                   help="Select a version")
    # parser_sync.add_argument("-u", "--uuid", dest="uuid",
    #                              nargs=1,
    #                              help="Select a uuid")


    parser_install = subpars_extn \
        .add_parser("install", help="Install an extension")
    parser_install.set_defaults(subparser_name="install")
    required_grp_install = parser_install.add_argument_group("required"
                                                       " arguments")
    required_grp_install.add_argument("-e", "--extension-name", dest="extn_name",
                                   required=True, nargs=1,
                                   help="Select an extension name")
    parser_install.add_argument("-d", "--target-file-path",
                                    dest="target_filepath",
                                    nargs=1,
                                    help="Select a target file path")
    parser_install.add_argument("-s", "--select-version",
                                   dest="select_version",
                                   nargs=1,
                                   help="Select a version")

    parser_versions = subpars_extn \
        .add_parser("versions",
                    help="Shows available versions for an extension")
    parser_versions.set_defaults(subparser_name="versions")
    required_grp_versions = parser_versions.add_argument_group(
        "required arguments")
    required_grp_versions.add_argument("-n", "--name", dest="name",
                                       required=True, nargs=1,
                                       help="Select an extension name")

    parser_variants = subpars_extn \
        .add_parser("variants",
                    help="Shows available variants for an extension")
    parser_variants.set_defaults(subparser_name="variants")
    required_grp_variants = parser_variants.add_argument_group(
        "required arguments")
    required_grp_variants.add_argument("-n", "--name", dest="name",
                                       required=True, nargs=1,
                                       help="Select an extension name")
    required_grp_variants.add_argument("-v", "--version", dest="vers",
                                       required=True, nargs=1,
                                       help="Select a version")

    parser_dependencies = subpars_extn \
        .add_parser("dependencies",
                    help="Shows all the dependencies for an extension")
    parser_dependencies.set_defaults(subparser_name="dependencies")
    required_grp_dependencies = parser_dependencies.add_argument_group(
        "required arguments")
    required_grp_dependencies.add_argument("-n", "--name", dest="name",
                                            required=True, nargs=1,
                                            help="Select an extension name")
    required_grp_dependencies.add_argument("-s", "--select-version", dest="select_version",
                                            required=True, nargs=1,
                                            help="Select a version")

def make_component_group(subpar):
    parser_comp = subpar.add_parser("comp", help="Perform actions on "
                                                 "components")
    parser_comp.set_defaults(parser_name="comp")
    subpars_comp = parser_comp.add_subparsers(metavar="")

    parser_info = subpars_comp \
        .add_parser("info",
                    help="Print component information including information "
                         "for all parameter in component")
    parser_info.set_defaults(subparser_name="info")
    required_grp_info = parser_info.add_argument_group("required arguments")
    required_grp_info.add_argument("-t", "--type", dest="type",
                                   required=True, nargs=1,
                                   help="Specify the component type")

    parser_list = subpars_comp.add_parser("list", help="List components")
    parser_list.set_defaults(subparser_name="list")
    parser_list.add_argument("-b", "--base-type", dest="base_type",
                             nargs=1, help="Filter components using base type")
    parser_list.add_argument("-r", "--repo-name", dest="repo_name",
                             nargs=1, help="Filter components using the "
                                           "repository name")
    parser_list.add_argument("-t", "--typename", dest="typename",
                             nargs=1, help="Filter components using the "
                                           "component type name")


def make_graph_group(subpar):
    parser_graph = subpar.add_parser("graph", help="Perform actions on graph")
    parser_graph.set_defaults(parser_name="graph")
    subpars_graph = parser_graph.add_subparsers(metavar="")

    parser_install = subpars_graph \
        .add_parser("install",
                    help="Install extensions for graph")
    parser_install.set_defaults(subparser_name="install")
    required_grp_install = parser_install.add_argument_group(
        "required arguments")
    required_grp_install.add_argument("-g", "--graph-files", dest="graph_files",
                                      required=True, nargs="+",
                                      help="Select one or more graph files")
    required_grp_install.add_argument("-m", "--manifest-file-path",
                                      dest="mnf_filepath",
                                      required=True, nargs=1,
                                      help="Select a manifest file path")
    required_grp_install.add_argument("-d", "--target-file-path",
                                      dest="target_filepath",
                                      required=True, nargs=1,
                                      help="Select a target file path")

    parser_install.add_argument("-u", "--output-directory",
                                dest="output_dir",
                                nargs=1,
                                help="Select an output directory to "
                                     "export graph contents")
    parser_install.add_argument("-r", "--archive-dir-path",
                                dest="archive_dirpath",
                                nargs=1,
                                help="Absolute path of directory to "
                                     " export the archive")
    parser_install.add_argument("-i", "--in-export-dir-path",
                                dest="in_export_dirpath",
                                nargs=1,
                                help="Directory structure to be used "
                                     "for export to directory / archive."
                                     " Default value in archive: /opt/nvidia/gxf/"
                                     " Default value in directory: \"\"")
    # parser_install.add_argument("-s", "--sm-arch",
    #                             dest="sm_arch",
    #                             nargs=1,
    #                             help="Select a sm architecture")

    parser_update = subpars_graph.add_parser("update",
                    help="Update extension versions to latest in graphs")
    parser_update.set_defaults(subparser_name="update")
    required_grp_update = parser_update.add_argument_group(
        "required arguments")
    required_grp_update.add_argument("-g", "--graph-files", dest="graph_files",
                                      required=True, nargs="+",
                                      help="Select one or more graph files")

def make_repo_group(subpar):
    parser_repo = subpar.add_parser("repo", help="Perform actions on "
                                                 "repositories")
    parser_repo.set_defaults(parser_name="repo")
    subpars_repo = parser_repo.add_subparsers(metavar="")

    parser_clean = subpars_repo.add_parser("clean", help="Clean the default repository")
    parser_clean.set_defaults(subparser_name="clean")

    parser_add = subpars_repo.add_parser("add", help="Add a repository")
    parser_add.set_defaults(subparser_name="add")

    parser_add_ngc = argparse.ArgumentParser(add_help=False)
    required_grp_add_ngc = parser_add_ngc.add_argument_group(
        "required arguments")
    required_grp_add_ngc.add_argument("-n", "--name", dest="name",
                                      required=True, nargs=1,
                                      help="Specify the name of the repository")
    required_grp_add_ngc.add_argument("-a", "--apikey", dest="apikey",
                                      required=True, nargs=1,
                                      help="Specify the API key")

    required_grp_add_ngc.add_argument("-o", "--org", dest="org",
                                      required=True, nargs=1,
                                      help="Specify the organization")

    parser_add_ngc.add_argument("-t", "--team", dest="team",
                                nargs=1,
                                help="Specify the team")

    subparsers_add = parser_add.add_subparsers(metavar="")

    parser_add_ngc.set_defaults(subsubparser_name="ngc")
    subparsers_add.add_parser("ngc", parents=[parser_add_ngc],
                              help="Add NGC repository")

    parser_remove = subpars_repo.add_parser("remove", help="Remove a "
                                                           "repository")
    parser_remove.set_defaults(subparser_name="remove")
    required_grp_rm = parser_remove.add_argument_group("required arguments")
    required_grp_rm.add_argument("-n", "--name", dest="name",
                                 required=True, nargs=1,
                                 help="Specify name of repository to be "
                                      "removed")

    parser_list = subpars_repo.add_parser("list", help="List of repositories")
    parser_list.set_defaults(subparser_name="list")
    parser_list.add_argument("-d", "--details", dest="details",
                             action="store_true",
                             help="Show details of each repository")

    parser_info = subpars_repo.add_parser("info", help="Provide information for a repositories")
    parser_info.set_defaults(subparser_name="info")
    parser_info.add_argument("-n", "--name", dest="name",
                             required=True, nargs=1,
                             help="Show details of a repository")

    parser_sync = subpars_repo \
        .add_parser("sync",
                    help="Scan the contents of specified repository and "
                         "sync cache")
    parser_sync.set_defaults(subparser_name="sync")
    parser_sync.add_argument("-n", "--name", dest="name",
                             required=True, nargs=1,
                             help="Specify name of repository to be synced")


def make_cache_group(subpar):
    parser_cache = subpar.add_parser("cache", help="Perform actions on "
                                                   "cache")
    parser_cache.set_defaults(parser_name="cache")
    group = parser_cache.add_mutually_exclusive_group()
    group.add_argument(
        "-s", "--set", dest="set",
        nargs=1, help="Set directory path for cache. If not set, then it uses "
                      "default path /var/tmp/gxf/.cache/gxf_registry")
    group.add_argument(
        "-c", "--clean", dest="clean", action="store_true",
        help="Clean registry cache, delete all cached information")
    group.add_argument(
        "-r", "--refresh", dest="refresh", action="store_true",
        help="Sync the latest versions of all the extensions from their source repository")
    group.add_argument(
        "-v", "--view", dest="view", action="store_true",
        help="View the current cache directory path")

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(parser_name="global")
    parser.add_argument("-v", "--version", dest="version",
                        action="store_true",
                        help="Print registry tool and GXF Spec version")
    subparsers = parser.add_subparsers(metavar="")
    make_cache_group(subparsers)
    make_repo_group(subparsers)
    make_component_group(subparsers)
    make_extension_group(subparsers)
    make_graph_group(subparsers)
    return parser
