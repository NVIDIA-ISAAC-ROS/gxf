# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Query maker
"""
import time
from typing import List
from result import Ok, Err, Result
from registry.core.core_interface import Registry
from registry.core.utils import Extension_Obj, Component_Obj


class QueryMaker:
    """ Query maker class, its purpose is to access
    elements from the extension list.
    """

    def __init__(self, registry):
        self.registry: Registry = registry
        self._extn_loaded = False
        self._comp_loaded = False
        self._constraint_exact_list = []
        self._constraint_in_list = []
        self._extn_list: List[Extension_Obj] = []
        self._comp_list: List[Component_Obj] = []
        self._repo_name = None

    def _load_data_extn(self) -> Result:
        if self._extn_loaded:
            return Ok("Extension list already loaded")
        res_ext_list = self.registry.get_extension_list(self._repo_name)
        if res_ext_list.is_err():
            return Err("Failed to retrieve extension list")
        self._extn_list = res_ext_list.value
        self._extn_loaded = True
        return Ok("Extension list successfully loaded")

    def _load_data_comp(self) -> Result:
        if self._comp_loaded:
            return Ok("Component list already loaded")
        extn_loaded = self._load_data_extn()
        if extn_loaded.is_err():
            return extn_loaded
        self._comp_list = []
        for i, extn in enumerate(self._extn_list):
            print(
                f'{i+1}/{len(self._extn_list)} => Fetching components for {extn.name}.')
            eid = extn.id
            res_comp_list = self.registry.get_component_list(eid)
            if res_comp_list.is_err():
                continue
            self._comp_list += res_comp_list.value
        return Ok("Components list successfully loaded")

    def drop_constraints(self) -> None:
        self._constraint_in_list = []
        self._constraint_exact_list = []

    @classmethod
    def _add_constraint_generic(cls, name, content, c_list):
        if isinstance(content, str):
            c_list.append([name, content])
        elif isinstance(content, (list, set)):
            for elm in content:
                c_list.append([name, elm])

    def add_constraint_exact_match_field(self, field_name: str,
                                         exact_content):
        self._add_constraint_generic(field_name, exact_content,
                                     self._constraint_exact_list)
        return self

    def add_constraint_inside_field(self, field_name: str,
                                    in_content):
        self._add_constraint_generic(field_name, in_content,
                                     self._constraint_in_list)
        return self

    @classmethod
    def test_keep_obj(cls, obj, constraint_list, func_test):
        for constraint in constraint_list:
            name = constraint[0]
            value = constraint[1]
            received_value = getattr(obj, name, "BAD_NAME")
            if received_value == "BAD_NAME":
                return Err("Attribute {} does not exist for extension"
                           .format(name))
            if not func_test(value, received_value):
                return Err("")
        return Ok("")

    def _get_generic(self, obj_list) -> Result:
        list_to_keep = []
        for extn in obj_list:
            keep_extn1 = self.test_keep_obj(extn, self._constraint_exact_list,
                                            lambda a, b: a == b)
            keep_extn2 = self.test_keep_obj(extn, self._constraint_in_list,
                                            lambda a, b: a in b)
            if keep_extn1.is_ok() and keep_extn2.is_ok():
                list_to_keep.append(extn)
        if not list_to_keep:
            return Err("Failed to find elements satisfying the provided "
                       "constraints\n{}".format(self))
        return Ok(list_to_keep)

    def set_repo_name(self, repo_name):
        res_repo_list = self.registry.get_repo_list()
        if res_repo_list.is_err():
            return res_repo_list
        repo_name_list = [elm.name for elm in res_repo_list.value]
        if repo_name not in repo_name_list:
            return Err("The selected repository does not exist")
        self._repo_name = repo_name
        return Ok()



    def get_for_extension(self) -> Result:
        res = self._load_data_extn()
        if res.is_err():
            return res
        res_extn_list = self._get_generic(self._extn_list)
        return res_extn_list

    def get_for_component(self) -> Result:
        res = self._load_data_comp()
        if res.is_err():
            return res
        res_comp_list = self._get_generic(self._comp_list)
        return res_comp_list

    def all_existing_attribute_components(self, attr_name):
        loaded = self._load_data_comp()
        if loaded.is_err():
            return loaded
        set_result = set()
        for elm in self._comp_list:
            received_value = getattr(elm, attr_name, "BAD_NAME")
            if received_value == "BAD_NAME":
                return Err("No attribute \"{}\" for components"
                           "".format(attr_name))
            set_result.add(received_value)
        return Ok(set_result)

    def all_existing_attribute_extensions(self, attr_name):
        loaded = self._load_data_extn()
        if loaded.is_err():
            return loaded
        set_result = set()
        for elm in self._extn_list:
            received_value = getattr(elm, attr_name, "BAD_NAME")
            if received_value == "BAD_NAME":
                return Err("No attribute \"{}\" for components"
                           "".format(attr_name))
            if isinstance(received_value, list):
                for val in received_value:
                    set_result.add(val)
            else:
                set_result.add(received_value)

        return Ok(set_result)

    def __str__(self):
        str_res = ""
        for elm in self._constraint_exact_list:
            str_res += " - {} : {}\n".format(elm[0], elm[1])
        for elm in self._constraint_in_list:
            str_res += " - {} : {}\n".format(elm[0], elm[1])
        return str_res
