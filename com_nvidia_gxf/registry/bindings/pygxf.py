#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

def gxf_context_create():
    return None


def get_runtime_version(context):
    return None


def gxf_context_destroy(context):
    return True


def gxf_load_extensions(context, path):
    return True


def gxf_load_extension_metadata(context, deps):
    return None

def get_ext_info(context, eid):
    return None

def get_comp_list(context, cid):
    return None

def get_comp_info(context, cid):
    return None

def get_param_list(context, cid):
    return None

def get_param_info(context, cid, key):
    return None
