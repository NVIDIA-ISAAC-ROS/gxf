# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Database Manager
"""

import base64
import hashlib
import os
import sqlite3
from packaging import version
import time
from typing import List

from registry.core.extension import Extension
from registry.core.repository import RegistryRepoConfig
from registry.core.utils import PlatformConfig, ComputeConfig, TargetConfig, ExtensionRecord, Variant
from registry.core.yaml_loader import YamlLoader
import registry.core.logger as log


TABLE_REPOSITORY_NAME = "repositories"
TABLE_REPOSITORY_CREATE = f"""
CREATE TABLE {TABLE_REPOSITORY_NAME} (
    name VARCHAR(255) NOT NULL,
    type VARCHAR(255),
    directory VARCHAR(1024),
    username VARCHAR(255),
    password VARCHAR(255),
    apikey VARCHAR(255),
    org VARCHAR(255),
    team VARCHAR(255),
    is_default_repo INTEGER,
    CONSTRAINT repo_unique UNIQUE (name, org, team)
);
"""

TABLE_INTERFACE_NAME = "interfaces"
TABLE_INTERFACE_CREATE = f"""
CREATE TABLE {TABLE_INTERFACE_NAME} (
    version VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    uuid VARCHAR(64) NOT NULL,
    repo_name VARCHAR(255) NOT NULL,
    metadata TEXT NOT NULL,
    hash VARCHAR(64) NOT NULL,
    frontend BOOLEAN NOT NULL CHECK (frontend in (0,1)),
    FOREIGN KEY (repo_name) REFERENCES {TABLE_REPOSITORY_NAME} (name),
    CONSTRAINT extn_unique_uuid UNIQUE (version, name, uuid, repo_name)
);
"""

TABLE_DEPENDENCIES_NAME = "dependency_matrix"
TABLE_DEPENDENCIES_CREATE = f"""
CREATE TABLE {TABLE_DEPENDENCIES_NAME} (
    extension_requiring_id INTEGER NOT NULL,
    extension_required_id INTEGER NOT NULL,
    FOREIGN KEY (extension_requiring_id) REFERENCES {TABLE_INTERFACE_NAME} (rowid)
    FOREIGN KEY (extension_required_id) REFERENCES {TABLE_INTERFACE_NAME} (rowid)
    CONSTRAINT extn_unique UNIQUE (extension_requiring_id, extension_required_id)
);
"""

TABLE_VARIANT_NAME = "variants"
TABLE_VARIANT_CREATE = f"""
CREATE TABLE {TABLE_VARIANT_NAME} (
    extension_id INT NOT NULL,
    repository_name VARCHAR(255) NOT NULL,
    gxf_core_version VARCHAR(64),
    registry_version VARCHAR(64),
    arch VARCHAR(255),
    os VARCHAR(255),
    distro VARCHAR(255),
    cuda VARCHAR(255),
    manifest TEXT,
    path VARCHAR(255),
    cudnn VARCHAR(255),
    tensorrt VARCHAR(255),
    triton VARCHAR(255),
    vpi VARCHAR(255),
    driver VARCHAR(255),
    deepstream VARCHAR(255),
    jetpack VARCHAR(255),
    gstreamer VARCHAR(255),
    hash VARCHAR(255),
    FOREIGN KEY (extension_id) REFERENCES {TABLE_INTERFACE_NAME}(rowid),
    FOREIGN KEY (repository_name) REFERENCES {TABLE_REPOSITORY_NAME}(name),
    CONSTRAINT variant_unique UNIQUE (extension_id, arch, os, distro, cuda, cudnn,
                                      tensorrt, deepstream, triton, vpi)
);
"""

logger = log.get_logger("Registry")

class DatabaseManager:

    def __init__(self, database_path: str):
        self._database_connection = None
        try:
            self._database_connection = sqlite3.connect(database_path)
            self._create_tables_if_missing()
        except sqlite3.Error as error:
            logger.error(f"Error while creating a sqlite table: {error}")

    def _perform_change_query(self, query, possible_error):
        if not self._database_connection:
            logger.error("Could not access to database")
            return False
        try:
            cur = self._database_connection.cursor()
            cur.execute(query)
            self._database_connection.commit()
        except sqlite3.Error as error:
            logger.debug(error)
            logger.error(possible_error)
            return False
        return True

    def _get_from_query(self, query, possible_error):
        if not self._database_connection:
            logger.error("Could not access to database")
            return None
        try:
            cur = self._database_connection.cursor()
            cur.execute(query)
            return cur.fetchall()
        except sqlite3.Error as error:
            logger.debug(error)
            logger.error(possible_error)
            return None

    def _create_tables_if_missing(self):
        cursor = self._database_connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        list_tables = [elm[0] for elm in cursor.fetchall()]
        if TABLE_REPOSITORY_NAME not in list_tables:
            cursor.execute(TABLE_REPOSITORY_CREATE)
        if TABLE_INTERFACE_NAME not in list_tables:
            cursor.execute(TABLE_INTERFACE_CREATE)
        if TABLE_DEPENDENCIES_NAME not in list_tables:
            cursor.execute(TABLE_DEPENDENCIES_CREATE)
        if TABLE_VARIANT_NAME not in list_tables:
            cursor.execute(TABLE_VARIANT_CREATE)
        cursor.close()

    def __del__(self):
        if self._database_connection:
            self._database_connection.close()

    def _get_table_content_as_str(self, table_name):
        c = self._database_connection.cursor()
        str_res = ""
        c.execute(f"PRAGMA table_info({table_name})")
        table_indexes = c.fetchall()
        c.execute(f"SELECT * FROM {table_name}")
        table_content = c.fetchall()

        max_size_content = [5] * (len(table_indexes) if table_indexes else 0)

        for row in table_content:
            for i, name in enumerate(list(row)):
                if isinstance(name, str):
                    max_size_content[i] = max(max_size_content[i], len(name) + 3)

        blob_indexes = []
        for i, name in enumerate(table_indexes):
            if name[2].lower() == "text":
                blob_indexes.append(i)
                continue
            if isinstance(name[1], str):
                max_size_content[i] = max(max_size_content[i], len(name[1]) + 3)
            str_res += f"| {name[1]}".ljust(max_size_content[i])
        str_res += "\n"

        for row in table_content:
            for i, name in enumerate(list(row)):
                if i in blob_indexes:
                    continue
                str_res += f"| {name}".ljust(max_size_content[i])
            str_res += "\n"
        str_res += "\n"
        return str_res

    def __str__(self):

        str_res = "DATABASE CONTENT:\n\n"
        str_res += self._get_table_content_as_str(TABLE_INTERFACE_NAME)
        str_res += self._get_table_content_as_str(TABLE_DEPENDENCIES_NAME)
        str_res += self._get_table_content_as_str(TABLE_REPOSITORY_NAME)
        str_res += self._get_table_content_as_str(TABLE_VARIANT_NAME)

        return str_res

    def update_repository_table(self, repo_list: list):
        if not self._database_connection:
            logger.error("Could not access to database")
            return False
        c = self._database_connection.cursor()
        for repo in repo_list:
            c.execute(f"SELECT * FROM \"{TABLE_REPOSITORY_NAME}\" WHERE name = \"{repo.name}\"")
            db_repo_list = c.fetchall()
            if not db_repo_list:
                self.add_repo(repo)
            else:
                db_repo = db_repo_list[0]
                if (repo.type == db_repo[1]
                        and repo.directory == db_repo[2]
                        and repo.username == db_repo[3]
                        and repo.password == db_repo[4]
                        and repo.apikey == db_repo[5]
                        and repo.org == db_repo[6]
                        and repo.team == db_repo[7]
                        and repo.default == db_repo[8]):
                    continue
                else:
                    query = f"""
                    UPDATE "{TABLE_REPOSITORY_NAME}"
                    SET type="{repo.type}", directory="{repo.directory}",
                    username="{repo.username}", password="{repo.password}",
                    apikey="{repo.apikey}", org="{repo.org}",
                    team="{repo.team}", is_default_repo="{repo.default}"
                    WHERE name="{repo.name}"
                    """
                    c.execute(query)

        c.execute(f"SELECT * FROM \"{TABLE_REPOSITORY_NAME}\"")
        db_repo_list = c.fetchall()
        names_bd = [repo[0] for repo in db_repo_list]
        names_repo_list = [repo.name for repo in repo_list]

        for name in names_bd:
            if name not in names_repo_list:
                self.remove_repo(name)
        return True

    def add_repo(self, repo_cfg: RegistryRepoConfig):
        query = f""" INSERT INTO {TABLE_REPOSITORY_NAME}
        (name, type, directory, username, password, apikey,
         org, team, is_default_repo)
         VALUES("{repo_cfg.name}", "{repo_cfg.type}", "{repo_cfg.directory}",
           "{repo_cfg.username}", "{repo_cfg.password}", "{repo_cfg.apikey}",
           "{repo_cfg.org}", "{repo_cfg.team}", "{repo_cfg.default}") """

        return self._perform_change_query(query, "Failed to add repository."
                                                 " This combination of (repository name, organization"
                                                 ", team)  already exists in cache")

    def remove_all_extensions(self, name):
        exts = self.get_extensions_from_repo(name)
        if exts is None:
            return False

        for ext in exts:
            if not self.remove_extension_from_database(ext):
                return False

        return True

    def remove_repo_if_exists(self, name):
        repos = self.get_all_repositories()
        if repos is None:
            return False

        if name in repos:
            return self.remove_repo(name)

        return True

    def remove_repo(self, name):

        if not self.remove_all_extensions(name):
            return False

        query = f"""
        DELETE FROM "{TABLE_REPOSITORY_NAME}" WHERE name="{name}";
        """
        return self._perform_change_query(query, "Failed to remove repository from cache")

    def drop_all_tables(self):
        return False

    def close_connection(self):
        if self._database_connection:
            self._database_connection.close()

    def add_interface_to_repo(self, repo_name: str, extension_metadata_path: str):
        if not os.path.exists(extension_metadata_path):
            logger.error(f"Metadata not found: {extension_metadata_path}")
            return False
        node = YamlLoader().load_yaml(extension_metadata_path)
        if not node:
            logger.error(f"bad metadata file: {extension_metadata_path}")
            return False
        ext_m: Extension = Extension.from_metadata(node)
        if not ext_m:
            return False
        ext = ExtensionRecord(ext_m.name, ext_m.version, ext_m.uuid)

        # Check if name exists in cache, if yes, verify the uuid matches.
        uuid = self.get_extension_uuid(ext.name)
        if uuid and uuid != ext.uuid:
            logger.error(f"Extension {uuid} is also registered with the same name")
            return False

        # Check if uuid exists in cache, if yes, verify the name matches.
        name = self.get_extension_name(ext.uuid)
        if name and name != ext.name:
            logger.error(f"Extension {name} is also registered with the same uuid")
            return False

        src_repo = self.get_extension_source_repo(ext)
        if src_repo:
            # If the same version of the extension is being added from the same or default repository,
            # remove the previous instance and then add the new one
            if repo_name == "default" or src_repo == repo_name:
                logger.debug(f"Removing stale extension {ext.name} from repo {src_repo}")
                if not self.remove_extension_from_database(ext):
                    return False
            else:
                logger.error(f"Same version of extension already exists in repository {src_repo}")
                return False

        e_metadata = None
        with open(extension_metadata_path, 'rb') as file:
            e_metadata = file.read()
        e_hash = hashlib.md5(e_metadata).hexdigest()
        base64_bytes = base64.b64encode(e_metadata)
        base64_string = base64_bytes.decode("ascii")

        # Check for frontend status
        is_frontend = 1
        fe_ver = self.get_frontend_version(ext.uuid)
        if fe_ver:
            if version.parse(ext.version) < version.parse(fe_ver):
                is_frontend = 0
            else:
                self.disable_frontend_flag(ext.uuid, fe_ver)

        query = f""" INSERT INTO {TABLE_INTERFACE_NAME}
                (version, name, uuid, repo_name, metadata, hash, frontend)
                VALUES("{ext.version}", "{ext.name}", "{ext.uuid}",
                "{repo_name}", "{base64_string}","{e_hash}", "{is_frontend}");
                """

        res = self._perform_change_query(query, "Failed to add interface."
                                                "This combination of (extension name, "
                                                "version, extension id, repo_name) already exists in cache")
        if not res:
            return False

        dependencies = [ExtensionRecord(self.get_extension_name(dep["uuid"]), dep["version"], dep["uuid"]) for dep in ext_m.dependencies]
        if not self.add_dependency(ext, dependencies):
            return False

        return self.update_dependencies()

    def remove_interface_from_repo(self, repo_name, ext_name, uuid, version):
        rowid = self._get_interface_rowid_from_ver_uuid_repo(version, uuid, repo_name)
        query = f""" SELECT * FROM "{TABLE_VARIANT_NAME}"
                 WHERE extension_id="{rowid}"
                 AND repository_name="{repo_name}";
                 """
        res = self._get_from_query(query, "Failed to read variants from cache")
        if res:
            logger.error("Failed to remove interface, please remove existing variants first")
            return False

        tuple_extn = ExtensionRecord(ext_name, version, uuid)
        if not self.remove_dependency(tuple_extn):
            return False

        query = f"""
                DELETE FROM "{TABLE_INTERFACE_NAME}" WHERE rowid="{rowid}";
                """
        if not self._perform_change_query(query, "Failed to remove interface from cache"):
            return False

        max_ver = self.get_latest_ext_version(ext_name, uuid)
        if max_ver and not self.update_frontend_flag(uuid, max_ver):
            return False

        return True

    def get_interface_metadata(self, ext: ExtensionRecord):
        query = f""" SELECT metadata  FROM {TABLE_INTERFACE_NAME}
                    WHERE name="{ext.name}" AND uuid="{ext.uuid}"
                    AND version="{ext.version}";
                 """
        res = self._get_from_query(query, "Failed to read extension metadata")
        if not res:
            logger.error("Extension does not exist in cache")
            return None
        base64_string = res[0][0]
        base64_bytes = base64_string.encode("ascii")
        sample_string_bytes = base64.b64decode(base64_bytes)
        metadata = sample_string_bytes.decode("ascii")

        return YamlLoader().load_string(metadata)

    def get_frontend_version(self, uuid):
        query = f""" SELECT DISTINCT version
                FROM {TABLE_INTERFACE_NAME}
                WHERE uuid="{uuid}" and frontend="1";
                """
        result = self._get_from_query(query, "Failed to read extensions info from cache")
        if not result:
            return None

        return result[0][0]

    def disable_frontend_flag(self, uuid, version):
        # Disable frontend flag for specific version
        query=f"""UPDATE {TABLE_INTERFACE_NAME} SET frontend="0"
                  WHERE uuid="{uuid}" and version="{version}" and frontend="1";
               """

        return self._perform_change_query(query, "Failed to update frontend version")

    def update_frontend_flag(self, uuid, version):
        # Update the frontend flag to new version
        current_fe = self.get_frontend_version(uuid)
        if current_fe and not self.disable_frontend_flag(uuid, current_fe):
            return False

        query=f"""UPDATE {TABLE_INTERFACE_NAME} SET frontend="1"
                  WHERE uuid="{uuid}" and version="{version}";
               """

        return self._perform_change_query(query, "Failed to update frontend version")

    def get_frontend_interfaces(self):
        # Read the latest version of the metadata for every extension
        query = f""" SELECT DISTINCT name, version, uuid FROM {TABLE_INTERFACE_NAME}
                     WHERE frontend="1";"""
        res = self._get_from_query(query, "Cache is corrupted")
        if res is None:
            return None

        frontend = []
        for r in res:
            record = ExtensionRecord(r[0], r[1], r[2])
            frontend.append(self.get_interface_metadata(record))

        if None in frontend:
            logger.error("Cache is corrupted")
            return None

        return frontend

    def _get_all_interfaces(self):
        # Read the metadata for every extension version and
        query = f""" SELECT DISTINCT metadata FROM {TABLE_INTERFACE_NAME}"""
        res = self._get_from_query(query, "Cache is corrupted")
        if res is None:
            return None

        result = []
        for interface in res:
            base64_string = interface[0]
            base64_bytes = base64_string.encode("ascii")
            string_bytes = base64.b64decode(base64_bytes)
            metadata = string_bytes.decode("ascii")

            result.append(YamlLoader().load_string(metadata))

        if None in result:
            logger.error("Cache is corrupted")
            return None

        return result

    def get_version_update(self, ext: ExtensionRecord):
        """ If any minor or patch updates have been done to the extension,
            return the latest version of that extension with the same major version
        """
        query = f"""SELECT version FROM "{TABLE_INTERFACE_NAME}" WHERE
                    uuid="{ext.uuid}" AND name="{ext.name}"
                 """
        result  = self._get_from_query(query, "Failed to find version update")
        if result is None:
            return None

        max_ver = version.parse(ext.version)
        for ver in result:
            cur_ver = version.parse(ver[0])
            if cur_ver.major == max_ver.major and cur_ver > max_ver:
                max_ver = cur_ver
                logger.info(f"Extension update found for extension: {ext.name}")
                logger.info(f"Old version: {ext.version} New version: {max_ver}")

        return ExtensionRecord(ext.name, str(max_ver), ext.uuid)

    def get_variant_hash(self, uuid, version, repo_name, target_cfg: TargetConfig):
        rowid = self._get_interface_rowid_from_ver_uuid_repo(version, uuid, repo_name)
        if not rowid:
            return None

        query = f""" SELECT hash from "{TABLE_VARIANT_NAME}" where extension_id="{rowid}"
                     AND arch="{target_cfg.platform.arch}"
                     AND os="{target_cfg.platform.os}"
                     AND distro="{target_cfg.platform.distribution}"
                     AND cuda="{target_cfg.compute.cuda}"
                     AND cuda="{target_cfg.compute.cuda}"
                     AND tensorrt="{target_cfg.compute.tensorrt}"
                     AND deepstream="{target_cfg.compute.deepstream}"
                     AND triton="{target_cfg.compute.triton}"
                     AND vpi="{target_cfg.compute.vpi}";
                """
        result  = self._get_from_query(query, "Failed to query extension variant hash")
        if not result:
            return None

        # Convert string None to NoneType
        if result[0][0] == "None":
            return None

        return result[0][0]

    def add_variant(self, uuid, version, repo_name,
                     variant: Variant) -> bool:
        rowid = self._get_interface_rowid_from_ver_uuid_repo(version, uuid, repo_name)
        if not rowid:
            return False

        # If variant is present for the same extension and source repo, do not add it again
        search_query = f""" SELECT * FROM "{TABLE_VARIANT_NAME}" WHERE extension_id="{rowid}"
                            AND arch="{variant.target.platform.arch}"
                            AND os="{variant.target.platform.os}"
                            AND distro="{variant.target.platform.distribution}"
                            AND cuda="{variant.target.compute.cuda}"
                            AND cudnn="{variant.target.compute.cudnn}"
                            AND tensorrt="{variant.target.compute.tensorrt}"
                            AND deepstream="{variant.target.compute.deepstream}"
                            AND triton="{variant.target.compute.triton}"
                            AND vpi="{variant.target.compute.vpi}";
                        """

        result  = self._get_from_query(search_query, "Failed to find extension variant")
        if result:
            return True

        query = f""" INSERT INTO {TABLE_VARIANT_NAME}
                    (extension_id, repository_name, gxf_core_version, registry_version,
                     arch, os, distro, cuda, cudnn, tensorrt, deepstream, triton, vpi, hash)
                    VALUES("{rowid}", "{repo_name}",
                           "{variant.gxf_core_version}",
                           "{variant.registry_version}",
                           "{variant.target.platform.arch}",
                           "{variant.target.platform.os}",
                           "{variant.target.platform.distribution}",
                           "{variant.target.compute.cuda}",
                           "{variant.target.compute.cudnn}",
                           "{variant.target.compute.tensorrt}",
                           "{variant.target.compute.deepstream}",
                           "{variant.target.compute.triton}",
                           "{variant.target.compute.vpi}",
                           "{variant.hash}")
                """

        return self._perform_change_query(
            query, f"Failed to add variant. This target config {variant.target} already exists in cache")

    def remove_variant(self, uuid, version, repo_name,
                        target: TargetConfig) -> bool:
        rowid = self._get_interface_rowid_from_ver_uuid_repo(version, uuid, repo_name)
        if not rowid:
            return False

        query = f""" DELETE FROM "{TABLE_VARIANT_NAME}" WHERE extension_id="{rowid}"
                        AND arch="{target.platform.arch}"
                        AND os="{target.platform.os}"
                        AND distro="{target.platform.distribution}"
                        AND cuda="{target.compute.cuda}"
                        AND cudnn="{target.compute.cudnn}"
                        AND tensorrt="{target.compute.tensorrt}"
                        AND deepstream="{target.compute.deepstream}"
                        AND triton="{target.compute.triton}"
                        AND vpi="{target.compute.vpi}";
                """
        return self._perform_change_query(query, "Failed to remove variant from cache")

    def get_variants(self, uuid, version, repo_name) -> List[Variant]:
        rowid = self._get_interface_rowid_from_ver_uuid_repo(version, uuid, repo_name)
        if not rowid:
            return None
        query = f"""SELECT arch, os, distro, cuda, cudnn, tensorrt, deepstream, triton, vpi,
                           gxf_core_version, registry_version, hash
                    FROM "{TABLE_VARIANT_NAME}" WHERE extension_id="{rowid}"
                    AND repository_name="{repo_name}"
        """
        variant_list = self._get_from_query(query, "Failed to read variants from cache")
        if not variant_list:
            return []
        config_list = []
        for variant in variant_list:
            variant = [var if var != "None" else None for var in variant]
            platform_cfg = PlatformConfig(variant[0], variant[1], variant[2])
            compute_cfg = ComputeConfig(variant[3], variant[4], variant[5], variant[6], variant[7], variant[8])
            target_cfg = TargetConfig(platform_cfg, compute_cfg)
            result = Variant(variant[9], variant[10], target_cfg, variant[11])
            config_list.append(result)
        return config_list

    def _get_interface_rowid_name_ver_uuid(self, ext: ExtensionRecord):
        query = f"""SELECT rowid FROM "{TABLE_INTERFACE_NAME}" WHERE
                                uuid="{ext.uuid}" AND version="{ext.version}"
                                AND name="{ext.name}"
                        """
        extn_list = self._get_from_query(query, "Failed to read interface from cache")
        if not extn_list:
            return None
        extn_rowid = extn_list[0][0]
        return extn_rowid

    def _get_interface_rowid_from_ver_uuid_repo(self, version, uuid, repo_name):
        query = f"""SELECT rowid FROM "{TABLE_INTERFACE_NAME}" WHERE
                                uuid="{uuid}" AND version="{version}"
                                AND repo_name="{repo_name}"
                        """
        extn_list = self._get_from_query(query, "Failed to read interface from cache")
        if extn_list is None:
            return None
        if len(extn_list) > 1:
            logger.error("Cache is corrupted")
            return None
        if len(extn_list) == 0:
            err_content = f"No matching interface with version: {version} uuid: {uuid}"
            if repo_name:
                err_content += f" repository: {repo_name}"
            logger.error(err_content)
            return None
        first_elm = extn_list[0]
        if not first_elm:
            logger.error("Cache is corrupted")  # if table is changed by user this could occur
            return None
        extn_rowid = first_elm[0]
        return extn_rowid

    def check_if_ext_exists(self, version, uuid, repo_name):
        rowid = self._get_interface_rowid_from_ver_uuid_repo(version, uuid, repo_name)
        if rowid:
            return True
        return False

    def add_dependency(self, ext: ExtensionRecord, dependencies: List[ExtensionRecord]):
        extn_rowid = self._get_interface_rowid_name_ver_uuid(ext)
        if not extn_rowid:
            logger.error(f"Failed to find extension: {ext.name} version: {ext.version}")
            return False

        for dep in dependencies:
            dep_rowid = self._get_interface_rowid_name_ver_uuid(dep)
            if not dep_rowid:
                logger.debug(f"Extension dependency not found {dep.name} version: {dep.version}")
                continue

            # Do not add dependency if already found in the database
            query = f""" SELECT * FROM {TABLE_DEPENDENCIES_NAME} WHERE
                         extension_requiring_id="{extn_rowid}" AND
                         extension_required_id="{dep_rowid}"
                     """
            if self._get_from_query(query, "Failed to search dependencies"):
                continue

            query = f""" INSERT INTO {TABLE_DEPENDENCIES_NAME}
            (extension_requiring_id, extension_required_id)
             VALUES("{extn_rowid}", "{dep_rowid}")
            """
            if not self._perform_change_query(
                    query, f"Failed to add dependencies. Dependencies already exist in cache"):
                logger.error(f"Failed to add dependency {dep.name}"
                                   f"to extension {ext.name}")
                return False

        return True

    def remove_dependency(self, ext: ExtensionRecord):
        extn_rowid = self._get_interface_rowid_name_ver_uuid(ext)
        if not extn_rowid:
            logger.error(f"Failed to find extension: {ext.name} version: {ext.version}")
            return False
        query = f""" DELETE FROM "{TABLE_DEPENDENCIES_NAME}"
                WHERE extension_requiring_id="{extn_rowid}"
                """
        return self._perform_change_query(query, "Failed to remove dependencies from cache")

    def get_dependencies(self, ext: ExtensionRecord) -> List[ExtensionRecord]:
        rowid = self._get_interface_rowid_name_ver_uuid(ext)
        if not rowid:
            logger.error(f"Failed to find extension: {ext.name} version: {ext.version}")
            return None
        query = f""" SELECT {TABLE_INTERFACE_NAME}.name,
                 {TABLE_INTERFACE_NAME}.version,
                 {TABLE_INTERFACE_NAME}.uuid
                FROM {TABLE_INTERFACE_NAME}
                INNER JOIN  {TABLE_DEPENDENCIES_NAME} TABDEP
                ON {TABLE_INTERFACE_NAME}.rowid=TABDEP.extension_required_id
                WHERE TABDEP.extension_requiring_id="{rowid}"
                """
        result = self._get_from_query(query, "Failed to read dependencies from cache")
        if result is None:
            return None

        return [ExtensionRecord(r[0], r[1], r[2]) for r in result]

    def update_dependencies(self):
        interfaces = self._get_all_interfaces()
        for i in interfaces:
            ext = Extension.from_metadata(i)
            if not ext:
                logger.error("Cache is corrupted")
                return False

            extn_tuple = ExtensionRecord(ext.name, ext.version, ext.uuid)
            dependencies = [ExtensionRecord(dep["extension"], dep["version"], dep["uuid"]) for dep in ext.dependencies]
            if not self.add_dependency(extn_tuple, dependencies):
                return False

        return True

    def get_extension_source_repo(self, ext: ExtensionRecord):
        query = f""" SELECT DISTINCT repo_name
                FROM {TABLE_INTERFACE_NAME}
                WHERE uuid="{ext.uuid}" AND version="{ext.version}" AND name="{ext.name}";
                """
        res = self._get_from_query(query, "Failed to read extension source repo from cache")
        if not res:
            return None
        if len(res) > 1:
            logger.error("Cache is corrupted")
            return None

        return res[0][0]

    def get_extension_name(self, uuid):
        query = f""" SELECT DISTINCT name
                FROM {TABLE_INTERFACE_NAME}
                WHERE uuid="{uuid}"
                """
        res = self._get_from_query(query, "Failed to read extension name")
        if not res:
            return None
        if len(res) > 1:
            logger.error("Cache is corrupted")
            return None

        return res[0][0]

    def get_extension_uuid(self, name):
        query = f""" SELECT DISTINCT uuid
                FROM {TABLE_INTERFACE_NAME}
                WHERE name="{name}"
                """
        res = self._get_from_query(query, "Failed to read extension uuid")
        if not res:
            return None
        if len(res) > 1:
            logger.error("Cache is corrupted")
            return None

        return res[0][0]

    def get_extensions_from_repo(self, repo_name):
        query = f""" SELECT DISTINCT name, version, uuid
                FROM {TABLE_INTERFACE_NAME}
                WHERE repo_name="{repo_name}";
                """
        result = self._get_from_query(query, "Failed to read extensions info from cache")
        if result is None:
            return None
        exts = [ExtensionRecord(r[0], r[1], r[2]) for r in result]
        return exts

    def get_ext_versions_from_repo(self, repo_name, extn_name, uuid):
        query = f""" SELECT DISTINCT version
                FROM {TABLE_INTERFACE_NAME}
                WHERE repo_name="{repo_name}" AND name="{extn_name}" AND uuid="{uuid}";
        """
        res = self._get_from_query(query, f"Failed to read versions of extension {extn_name} from {repo_name}")
        if not res:
            return None
        return [elm[0] for elm in res]

    def get_all_ext_versions(self, extn_name, uuid):
        """ Returns a dictionary with repo names as keys and a list of versions as values
        """
        query = f""" SELECT DISTINCT version, repo_name
                FROM {TABLE_INTERFACE_NAME}
                WHERE name="{extn_name}" AND uuid="{uuid}";
        """
        res = self._get_from_query(query, f"Failed to read versions of extension {extn_name}")
        if not res:
            return None

        version_dict = {}
        for r in res:
            repo = r[1]
            ver = r[0]
            if repo not in version_dict.keys():
                version_dict[repo] = [ver]
            else:
                version_dict[repo].append(ver)

        return version_dict

    def get_latest_ext_version(self, extn_name, uuid):
        query = f""" SELECT DISTINCT version
                FROM {TABLE_INTERFACE_NAME}
                WHERE name="{extn_name}" AND uuid="{uuid}";
        """
        res = self._get_from_query(query, "Failed to read latest extension version")
        if not res:
            return None

        versions = [version.parse(elm[0]) for elm in res]
        max_ver = versions[0]
        for ver in versions:
            max_ver = ver if ver > max_ver else max_ver

        return str(max_ver)

    def get_repositories_from_extension_name(self, extn_name):
        query = f""" SELECT DISTINCT repo_name
                        FROM {TABLE_INTERFACE_NAME}
                        WHERE name="{extn_name}";
                """
        res = self._get_from_query(query, "Failed to read extension source repositories")
        if not res:
            return None
        return [elm[0] for elm in res]

    def remove_extension_from_database(self, ext: ExtensionRecord):
        logger.debug(f"Removing extension from cache {ext}")
        src_repo = self.get_extension_source_repo(ext)

        variants = self.get_variants(ext.uuid, ext.version, src_repo)
        for var in variants:
            if not self.remove_variant(ext.uuid, ext.version, src_repo, var.target):
                return False

        if not self.remove_interface_from_repo(src_repo, ext.name, ext.uuid, ext.version):
            return False

        return True

    def get_all_repositories(self):
        query = f""" SELECT DISTINCT repo_name
                        FROM {TABLE_INTERFACE_NAME};
                 """

        return self._get_from_query(query, "Failed to read repositories")