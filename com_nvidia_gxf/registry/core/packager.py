# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Registry Tarball Packager
"""

import tarfile
import os, os.path as path
from os.path import expanduser, abspath

from registry.core.utils import compute_sha256
import registry.core.logger as log

logger = log.get_logger("Registry")
class Packager():
    def __init__(self, archive_dirpath: str, name: str = None):
        abspath = self._cleanpath(archive_dirpath)
        self._makedirs(abspath)
        self._package_name = str("package" if not name else name) + ".tar.gz"
        self._package_path = path.join(abspath, self._package_name)
        self._tarball = None

        try:
          self._tarball = tarfile.open(self._package_path, mode="w:gz")
        except tarfile.TarError:
          logger.error(f"Failed to create tarfile {abspath}")

    @property
    def package_path(self):
        return self._package_path

    @property
    def package_name(self):
        return self._package_name

    def zip(self):
        self._tarball.close()
        logger.debug(f"Tarball created at : {self._package_path}")

    def addDirectory(self, dir_path: str,  arc_path: str = "", exclude_files = []):

        abspath = self._cleanpath(dir_path)
        if not path.isdir(abspath):
            logger.error(f"Not a valid directory: {abspath}")
            return False

        self._tarball.add(abspath, arcname=arc_path, filter=lambda tarinfo: None if os.path.splitext(tarinfo.name)[1] in exclude_files else tarinfo)
        return True

    def addFile(self, file_path: str, arc_dir: str):
        abspath = self._cleanpath(file_path)
        if not os.path.isfile(abspath):
            logger.error(f"Missing file {abspath}")
            return False
        arc_path = path.join(arc_dir, path.split(abspath)[-1])
        self._tarball.add(abspath, arc_path)
        return True

    def _cleanpath(self, i_path: str):
        abspath = path.abspath(expanduser(i_path))
        return abspath

    def _makedirs(self, abspath: str):
        if os.path.exists(abspath):
            return abspath
        elif os.path.isfile(abspath):
            return os.path.dirname(abspath)
        else:
            os.makedirs(abspath, mode=0o777, exist_ok=True)
        return abspath

    def sha256_hash(self):
        if not os.path.isfile(self._package_path):
            logger.error(f"Missing file {self._package_path}")
            return None

        return compute_sha256(self._package_path)