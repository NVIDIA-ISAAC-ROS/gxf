# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from enum import Enum, unique, IntEnum, IntFlag
import hashlib
import logging
import re
from typing import NamedTuple, Any

import registry.core.logger as log

logger = log.get_logger("Registry")

BUF_SIZE = 1024 * 1024 * 2  # 2mb

# const struct like objects to communicate with composer and cli tool
class Extension_Obj(NamedTuple):
    id: str
    name: str
    desc: str
    version: str
    license: str
    author: str
    labels: list
    display_name: str
    category: str
    brief: str


class Parameter_Obj(NamedTuple):
    key: str
    headline: str
    desc: str
    flags: str
    parameter_type: str
    default: Any
    handle_type: str
    rank: int
    shape: list
    min_value: Any
    max_value: Any
    step_value: Any


class Component_Obj(NamedTuple):
    cid: str
    typename: str
    base_typename: str
    is_abstract: bool
    desc: str
    display_name: str
    brief: str

# Tuple to uniquely identify an extension
class ExtensionRecord(NamedTuple):
    name: str
    version: str
    uuid: str

class PlatformConfig(NamedTuple):
    arch: str
    os: str
    distribution: str

class ComputeConfig(NamedTuple):
    cuda: str
    cudnn: str
    tensorrt: str
    deepstream: str
    triton: str
    vpi: str

class TargetConfig(NamedTuple):
    platform: PlatformConfig
    compute: ComputeConfig

class Variant(NamedTuple):
    gxf_core_version: str
    registry_version: str
    target: TargetConfig
    hash: str
    # priority: int

def uuid_validator(uuid: str):
    result = re.fullmatch("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                          "-[0-9a-f]{4}-[0-9a-f]{12}", uuid)
    if not result:
        logger.error("UUID must conform to standard IETF format specified in https://tools.ietf.org/html/rfc4122")
        logger.error(f"{uuid} does not match the requirement")
        return None

    return uuid

def semantic_version_validator(version: str):
    result = re.fullmatch("^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$", version)
    if not result:
        logger.error("Version must conform to standard semantic versioning format specified in https://semver.org/")
        logger.error(f"{version} does not match the requirement")
        return None

    return version

def extension_name_validator(name: str):
    result = re.fullmatch("^(?=.{1,256}$)[a-zA-Z][a-zA-Z\d_]*$", name)
    if not result:
        logger.error("Extension name must conform to format \"^(?=.{1,256}$)[a-zA-Z][a-zA-Z\d_]*$\"")
        logger.error(f"{name} does not match the requirement")
        return None

    return name

def priority_validator(priority: str):
    result = re.fullmatch("^[0-9][0-9]?$|^100$", priority)
    if not result:
        logger.error("Extension priority must be a number between 0-100")
        logger.error(f"{priority} does not match the requirement")
        return None

    return priority

def get_ext_subdir(eid, version, target: TargetConfig):
    distro = target.platform.distribution if target.platform.distribution else "undefined"
    arch = target.platform.arch if target.platform.arch else "undefined"
    opsys = target.platform.os if target.platform.os else "undefined"

    compute_str = ""
    compute_str  += f"-cuda-{target.compute.cuda}" if target.compute.cuda else ""
    compute_str  += f"-cudnn-{target.compute.cudnn}" if target.compute.cudnn else ""
    compute_str  += f"-trt-{target.compute.tensorrt}" if target.compute.tensorrt else ""
    compute_str  += f"-ds-{target.compute.deepstream}" if target.compute.deepstream else ""
    compute_str  += f"-triton-{target.compute.triton}" if target.compute.triton else ""
    compute_str  += f"-vpi-{target.compute.vpi}" if target.compute.vpi else ""

    ext_default_path = f"{eid}/{version}/{opsys}/{arch}/{distro}{compute_str}"
    return ext_default_path.replace(" ","_").lower()

def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def target_to_str(cfg: TargetConfig):
    res = f"arch: {cfg.platform.arch} os: {cfg.platform.os} distribution: {cfg.platform.distribution} "

    res += f"cuda: {cfg.compute.cuda} " if cfg.compute.cuda else ""
    res += f"cudnn: {cfg.compute.cudnn} " if cfg.compute.cudnn else ""
    res += f"tensorrt: {cfg.compute.tensorrt} " if cfg.compute.tensorrt else ""
    res += f"deepstream: {cfg.compute.deepstream} " if cfg.compute.deepstream else ""
    res += f"triton: {cfg.compute.triton} " if cfg.compute.triton else ""
    res += f"vpi: {cfg.compute.vpi} " if cfg.compute.vpi else ""

    return res
