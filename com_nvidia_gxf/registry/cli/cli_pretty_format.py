# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Pretty format module
"""

import textwrap

from registry.core.config import RegistryRepoConfig
from registry.core.repository import RepositoryType
from registry.core.utils import Component_Obj, Extension_Obj, \
    Parameter_Obj, ExtensionRecord
from result import Ok, Result
from typing import List


def format_name_desc(name: str, desc: str) -> str:
    """ formats a long description to wrap it
    """
    space_len = len(name) + 4
    space = space_len * " "
    wrapped = textwrap.fill(desc, width=100 - space_len)
    list_desc_split = wrapped.split("\n")
    tmp_str = ""
    for i, elm in enumerate(list_desc_split):
        if i == 0:
            tmp_str += name + " :  " + elm
        else:
            tmp_str += space + elm
        if i != len(list_desc_split) - 1:
            tmp_str += "\n"
    return tmp_str


def wrap_with_padding(len_padding: int, key: str,
                      text: str, len_text: int,
                      skip_first_space=False,
                      split_char=" : ") -> str:
    """ formats a long description to wrap it and add padding to it
    :param len_padding: number of spaces to pad
    :param key: title of text to be displayed
    :param text: content to be wrapped
    :param len_text: expected length of a line
    :param skip_first_space: removes
    :param split_char: character that will be put between key and text
    :return:
    """
    space = len_padding * " "
    text_full = "{}{}{}".format(key, split_char, text)
    wrapped_t = textwrap.fill(text_full, width=len_text)
    str_acc = ""
    list_wrapped_t = wrapped_t.split("\n")
    for i, elm in enumerate(list_wrapped_t):
        if i == 0 and skip_first_space:
            str_acc += elm
        else:
            str_acc += "{}{}".format(space, elm)
        if i != len(list_wrapped_t) - 1:
            str_acc += "\n"
    return str_acc


def format_parameter_for_printing(param: Parameter_Obj) -> str:
    """ Returns a str containing a parameter formatted for display.
    """
    space_key = 6
    len_to_wrap = 80 - space_key
    msa = 19  # max size attribute
    key_len = len(param.key)
    space_to_add = space_key - key_len - 2 + msa
    space_after_key = space_to_add if space_to_add > 0 else 0
    tmp_str = "  " + param.key + (space_after_key * " ")
    # wrapper around a text formatter
    fkt = lambda k, t: wrap_with_padding(space_key, k, t, len_to_wrap)
    tmp_str += wrap_with_padding(space_key, "", param.headline,
                                 len_to_wrap, True) + "\n"
    if param.desc:
        desc_name = (" " * 6) + "description".ljust(msa)
        tmp_str += format_name_desc(desc_name, param.desc) + "\n"
    tmp_str += fkt("flags".ljust(msa), param.flags) + "\n"
    tmp_str += fkt("parameter_type".ljust(msa), param.parameter_type) + "\n"
    tmp_str += fkt("default".ljust(msa), str(param.default)) + "\n"
    tmp_str += fkt("rank".ljust(msa), str(param.rank)) + "\n"
    tmp_str += fkt("shape".ljust(msa), (' '.join([str(elem) for elem in param.shape])) if param.shape else "None") + "\n"
    tmp_str += fkt("min_value".ljust(msa), str(param.min_value)) + "\n"
    tmp_str += fkt("max_value".ljust(msa), str(param.max_value)) + "\n"
    tmp_str += fkt("step_value".ljust(msa), str(param.step_value))
    if param.parameter_type == "GXF_PARAMETER_TYPE_HANDLE":
        tmp_str += "\n" + fkt("handle_type".ljust(msa), param.handle_type)
    return tmp_str


def format_component(component: Component_Obj) -> str:
    tmp_str = component.typename + " :\n"
    descc = component.desc
    if not descc:
        descc = "N/A"
    tmp_str += wrap_with_padding(4, "", descc, 76, False, "")
    return tmp_str


def format_component_with_details(component: Component_Obj, shift_right=0) \
        -> str:
    """ Returns a str containing a component formatted for display.
    """
    formatted_str = ""
    sp = "" + (" " * shift_right)
    cp_typename = component.typename if component.typename else "N/A"
    formatted_str += "{}typename :      {}\n".format(sp, cp_typename)
    cp_id = component.cid if component.cid else "N/A"
    formatted_str += "{}cid :           {}\n".format(sp, cp_id)
    cp_basetypename = component.base_typename if component.base_typename else "N/A"
    formatted_str += "{}base typename : {}\n".format(sp, cp_basetypename)
    formatted_str += "{}is abstract :   {}".format(sp, component.is_abstract)
    if component.brief:
        formatted_str += "\nbrief :         {}".format(sp, component.brief)
    if component.desc:
        formatted_str += "\ndescription :   "
        formatted_str += wrap_with_padding(16, "", component.desc,
                                           64, True, "")
    return formatted_str


def format_extension(extension: Extension_Obj) -> str:
    """ Returns a str containing an extension formatted for display.
    """
    desc = "{} : {}\n".format(extension.name, extension.desc)
    formatted_str = textwrap.fill(desc, width=100)
    return formatted_str


def format_extension_with_details(extension: Extension_Obj,
                                  shift_right=0) -> str:
    """ Returns a str containing an extension formatted for display.
    """
    formatted_str = ""
    sp = "" + (" " * shift_right)
    formatted_str += "{}name :        {}\n".format(sp, extension.name)
    formatted_str += "{}uuid :        {}\n".format(sp, extension.id)
    formatted_str += "{}brief :       {}\n".format(sp, extension.brief)
    desc = "description : {}\n".format(extension.desc)
    desc_w = textwrap.fill(desc, width=100 - shift_right)
    for line in desc_w.split("\n"):
        formatted_str += "{}{}\n".format(sp, line)

    formatted_str += "{}version :     {}\n".format(sp, extension.version)
    formatted_str += "{}license :     {}\n".format(sp, extension.license)
    formatted_str += "{}author :      {}\n".format(sp, extension.author)
    formatted_str += "{}category :    {}".format(sp, extension.category)
    nb_lbl = 0
    if isinstance(extension.labels, list):
        nb_lbl = len(extension.labels)
    if nb_lbl != 0:
        formatted_str += "\n"
        formatted_str += "{}labels:\n".format(sp)
        for label in extension.labels:
            formatted_str += "{}  - {}\n".format(sp, label)

    return formatted_str


class PrettyFormatter:
    """ Class containing class methods to format data as a string.
    Each method shall return a Result object containing a string
    describing the action performed of the error encountered.
    """

    @classmethod
    def versions(cls, result: Result) -> Result:
        """ Get registry and GXF core versions
        :param result: Result containing a tuple with core and registry
        versions
        :return:
        """
        if result.is_err():
            return result
        core_v, reg_v = result.value
        str_acc = "Registry Version : {}\n".format(reg_v.value)
        str_acc += "GXF Core Version : {}".format(core_v.value)
        return Ok(str_acc)

    @classmethod
    def extension_list(cls, result: Result) -> Result:
        """ Format a list of extension to a string
        :param result: Result containing a list of extension to display
        :return: formatted string
        """
        if result.is_err():
            return result
        ext_list: List[Extension_Obj] = result.value
        ext_list.sort(key=lambda repo: repo.name.lower())
        str_acc = ""
        for i, elm in enumerate(ext_list):
            str_acc += format_extension(elm)
            if i != len(ext_list) - 1:
                str_acc += "\n"
        return Ok(str_acc)

    @classmethod
    def extension_info(cls, result_extn: Result,
                       result_comp_list: Result) -> Result:
        """ Format the information about an extension to a string
        :param result_extn: Result containing an extension to display
        :param result_comp_list: Result containing a list of
         components to display
        :return: formatted string
        """
        if result_extn.is_err():
            return result_extn
        str_acc = format_extension_with_details(result_extn.value)
        if result_comp_list.is_ok():
            str_acc += "\n"
            str_acc += "components :"
            if result_comp_list.value:
                str_acc += "\n" + cls.component_list(result_comp_list).value
            else:
                str_acc += "  N/A"
        return Ok(str_acc)

    @classmethod
    def component_list(cls, result_comp_list: Result) -> Result:
        """ Format a list of components to a string
        :param result_comp_list:
        :return: formatted string
        """
        if result_comp_list.is_err():
            return result_comp_list
        comp_list = result_comp_list.value
        comp_list.sort(key=lambda repo: repo.typename.lower())
        str_acc = ""
        for i, obj in enumerate(comp_list):
            str_acc += format_component(obj)
            if i != len(comp_list) - 1:
                str_acc += "\n"
        return Ok(str_acc)

    @classmethod
    def component_info(cls, result_comp: Result, result_param_list: Result) \
            -> Result:
        """ Format the information about an component to a string
        :param result_comp: Result containing a component to display
        :param result_param_list: list of parameters to display
        :return: formatted string
        """
        if result_comp.is_err():
            return result_comp

        obj: Component_Obj = result_comp.value
        str_acc = format_component_with_details(obj, 0)
        if result_param_list.is_ok():
            str_acc += "\nParameters :"
            if result_param_list.value:
                str_acc += "\n"
                for i, elm in enumerate(result_param_list.value):
                    str_acc += format_parameter_for_printing(elm)
                    if i != len(result_param_list.value) - 1:
                        str_acc += "\n"
            else:
                str_acc += "    N/A"
        return Ok(str_acc)

    @classmethod
    def string_list(cls, strlist: List[str]) -> str:
        """ Formats a list of string for printing
        :param strlist: the list of string to be formatted
        :return: the formatted string
        """
        strlist.sort()
        str_res = ""
        for i, elm in enumerate(strlist):
            if elm == "":
                continue
            str_res += " - {}".format(elm)
            if i != len(strlist) - 1:
                str_res += "\n"
        return str_res

    @classmethod
    def filtering_possibilities(cls, listing_fun, attr_name) -> str:
        """ Format a string which will display possibilities for
         an attribute
        :param listing_fun: the function that will be used to generate a list
        for the selected attribute
        :param attr_name: name of the attribute
        :return: the formatted string
        """
        res_all = listing_fun(attr_name)
        if res_all.is_err():
            return res_all.value
        list_val = list(res_all.value)
        str_err = ""
        if list_val:
            str_err = f"\nPlease choose among the following {attr_name}:\n"
            list_res = list(res_all.value)
            str_err += PrettyFormatter.string_list(list_res)
        return str_err

    @classmethod
    def repository(cls, repo: RegistryRepoConfig, details: bool = False,
                   single_repo=False) -> str:
        """ Format the content of a repository object in a string
        :param repo: the repository to be formatted
        :param details: allows more details to be provided
        :return: the formatted string
        """
        res_str = ""
        if not single_repo:
            res_str += "\n"
        msa = 15
        if details:
            res_str += wrap_with_padding(0, "- " + repo.name, "", 80)

            if repo.type == "local":
                res_str += "\n"
                res_str += wrap_with_padding(4, "type".ljust(msa),
                                             RepositoryType.LOCAL.value, 76)
                res_str += "\n"
                res_str += wrap_with_padding(4, "directory paths".ljust(msa),
                                             repo.directory, 76)
            elif repo.type == "ngc":
                res_str += "\n"
                res_str += wrap_with_padding(4, "type".ljust(msa),
                                             RepositoryType.NGC.value, 76)
                if repo.org:
                    res_str += "\n"
                    res_str += wrap_with_padding(4, "org".ljust(msa),
                                                 repo.org, 76)
                if repo.team:
                    res_str += "\n"
                    res_str += wrap_with_padding(4, "team".ljust(msa),
                                                 repo.team, 76)

        else:
            res_str += "- " + repo.name
        return res_str

    @classmethod
    def repository_list(cls, repositories, details):
        """ Formats a list of repositories
        :param repositories: the list of repositories
        :param details: allows more details to be provided
        :return: the formatted string
        """
        str_acc = "Repositories:"
        repositories.sort(key=lambda repo: repo.name.lower())
        for repo in repositories:
            str_acc += cls.repository(repo, details)
        return str_acc

    @classmethod
    def variant_list(cls, variant_list):
        result = []
        for variant in variant_list:
            cfg = variant.target
            str = f"arch: {cfg.platform.arch} os: {cfg.platform.os} distro: {cfg.platform.distribution} "
            if cfg.compute.cuda:
                str += f"cuda: {cfg.compute.cuda} "
            if cfg.compute.cudnn:
                str += f"cudnn: {cfg.compute.cudnn} "
            if cfg.compute.tensorrt:
                str += f"tensorrt: {cfg.compute.tensorrt} "
            if cfg.compute.deepstream:
                str += f"deepstream: {cfg.compute.deepstream} "
            if cfg.compute.triton:
                str += f"triton: {cfg.compute.triton} "
            if cfg.compute.vpi:
                str += f"vpi: {cfg.compute.vpi} "
            if variant.gxf_core_version:
                str += f"gxf core version: {variant.gxf_core_version}"
            result.append(str)
        return cls.string_list(result)

    @classmethod
    def extension_dependencies(cls, extension_list:List[ExtensionRecord]):
        res = [f"name: {ext.name} version: {ext.version}" for ext in extension_list]
        return cls.string_list(res)

    @classmethod
    def dict_of_string_list(cls, str_list: dict) -> str:
        """ Formats a dict with strings as key and list of strings as values
        :param str_list: the list of string to be formatted
        :return: the formatted string
        """
        str_res = ""
        for k, v in str_list.items():
          val_str = ""
          for item in v:
              val_str += item + ", "
          val_str = val_str.rstrip(", ")
          str_res += wrap_with_padding(0, k, val_str, 40) + "\n"
        return str_res.rstrip()
