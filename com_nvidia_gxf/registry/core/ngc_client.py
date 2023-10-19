# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" NGC Client for Registry
"""

import base64
import datetime
import json
from multiprocessing import cpu_count
import os, os.path as path
from os.path import expanduser, abspath
import requests
from requests.adapters import HTTPAdapter, Retry
from packaging import version
from zipfile import ZipFile
import shutil
import sys
import tarfile
from tarfile import TarFile
import tempfile
import toml
from urllib.error import HTTPError
import yaml

from registry.core.config import NGC_NO_API_KEY, NGC_NO_ORG, NGC_NO_TEAM
from registry.core.extension import Extension
from registry.core.packager import Packager
from registry.core.utils import TargetConfig, uuid_validator, get_ext_subdir
from registry.core.yaml_loader import YamlLoader
import registry.core.logger as log

NGC_AUTH_TOKEN_TIMEOUT = 10    # Second

logger = log.get_logger("Registry")
class PublicNGCClient:
    """Implementation of an NGC Client which interacts with public NGC registry
    """

    def __init__(self):

        self._v2_url = "https://api.ngc.nvidia.com/v2"
        self._session = requests.session()
        retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[ 500, 502, 503, 504 ])
        self._session.mount("https://", HTTPAdapter(max_retries=retries,pool_connections=cpu_count(), pool_maxsize=cpu_count()))

    def _get_ext_primary_list_url(self):
        return self._v2_url + "/resources/nvidia/graph-composer/manifest/versions/primarymanifest/files/manifest.toml"

    def _get_ext_secondary_list_url(self):
        return self._v2_url + "/resources/nvidia/graph-composer/manifest/versions/secondarymanifest/files/manifest.toml"

    def _get_ext_test_list_url(self):
        return self._v2_url + "/resources/nvidia/graph-composer/manifest/versions/testmanifest/files/manifest.toml"

    def _get_ext_download_url(self, ext_name, ext_version):
        ext_name = ext_name.lower()
        return self._v2_url + f"/resources/nvidia/graph-composer/{ext_name}/versions/{ext_version}/zip"

    def _get_ext_file_download_url(self, ext_name, ext_version, filename):
        ext_name = ext_name.lower()
        filename = filename.lower()
        return self._v2_url + f"/resources/nvidia/graph-composer/{ext_name}/versions/{ext_version}/files/{filename}"

    def _get_ext_version_url(self, ext_name):
        ext_name = ext_name.lower()
        return self._v2_url + f"/resources/nvidia/graph-composer/{ext_name}/versions"

    def _get_request_headers(self):
        headers = {
            'Content-Type': "application/json",
            'Accept': "application/json",
            'Cache-Control': "no-cache",
            'Host': "api.ngc.nvidia.com",
            'Accept-Encoding': "gzip, deflate",
            'Connection': "keep-alive",
            'cache-control': "no-cache"
        }
        return headers

    def _get_download_headers(self, ext_url):
        headers = {
            'Accept': '*/*',
            'Cache-Control': 'no-cache',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': f"{ext_url}",
            'Connection': 'keep-alive',
            'cache-control': 'no-cache'
        }
        return headers

    def _get_variant_ver_str(self, version, target_cfg: TargetConfig):
        distro = target_cfg.platform.distribution
        arch = target_cfg.platform.arch
        opsys = target_cfg.platform.os

        compute_str = ""
        compute_str  += f"-cuda-{target_cfg.compute.cuda}" if target_cfg.compute.cuda else ""
        compute_str  += f"-cudnn-{target_cfg.compute.cudnn}" if target_cfg.compute.cudnn else ""
        compute_str  += f"-trt-{target_cfg.compute.tensorrt}" if target_cfg.compute.tensorrt else ""
        compute_str  += f"-ds-{target_cfg.compute.deepstream}" if target_cfg.compute.deepstream else ""
        compute_str  += f"-triton-{target_cfg.compute.triton}" if target_cfg.compute.triton else ""
        compute_str  += f"-vpi-{target_cfg.compute.vpi}" if target_cfg.compute.vpi else ""

        version_str = f"{version}-{opsys}-{arch}-{distro}{compute_str}"
        return version_str.replace(" ", "_").lower()

    def get_extension_versions(self, ext_name):
        ext_name = ext_name.lower()

        version_url = self._get_ext_version_url(ext_name)
        if version_url is None:
            logger.warning("Failed to create version url")
            return None

        headers = self._get_request_headers()
        if headers is None:
            return None

        query_string = {'scope': 'group/ngc:nvidia'}
        response = self._session.get(version_url, headers=headers, params=query_string)

        if response.status_code != 200:
            logger.error(f"Extension version request failed for {ext_name}")
            self._log_error_response(response)
            return None

        payload = json.loads(response.text)
        if "recipeVersions" not in payload:
            logger.warning(f"No extension versions found for extension {ext_name}")
            return None

        versions = [ver["versionId"] for ver in payload["recipeVersions"]]
        return versions

    def get_extension_list(self):
        import_dp =  tempfile.mkdtemp(prefix=f"{tempfile.gettempdir()}//.nvgraph.")

        def download_manifest(url):
            headers = self._get_download_headers(url)
            return self._download_file(url, headers, import_dp)

        ext_url = self._get_ext_primary_list_url()
        result = download_manifest(ext_url)
        if not result:
            ext_url = self._get_ext_secondary_list_url()
            result = download_manifest(ext_url)
            if not result:
                logger.error("Extension download from Public NGC Catalog failed")
                return None

        try:
            manifest_file = toml.load(path.join(import_dp,"manifest.toml"))
        except toml.decoder.TomlDecodeError as e:
            logger.error("Extension download from Public NGC Catalog failed")
            logger.debug(f"{e}")
            return None

        extn_list= {}
        for ext in manifest_file["extension"]:
            if uuid_validator(ext["eid"]):
                extn_list[ext["eid"]] = ext["name"]

        try:
            shutil.rmtree(import_dp)
        except IOError:
            logger.debug(f"Failed to remove {import_dp}")

        return extn_list

    def _extract_tar_file(self, filepath, extract_path):
        with TarFile.open(filepath, mode="r:gz") as tarball:
            logger.debug(f"Extracting file {filepath} to {extract_path}")
            for tarf in tarball:
                try:
                    tarball.extract(tarf.name, extract_path)
                except IOError:
                    logger.warning(f"Failed to extract tarfile {tarf.name}."
                                        " Possible duplicate, removing the old file")
                    os.remove(path.join(extract_path, tarf.name))
                    tarball.extract(tarf.name, extract_path)

    def _extract_zip_file(self, filepath, extract_path):
        with ZipFile(filepath) as zipf:
            logger.debug(f"Extracting zip file {filepath}")
            contents = zipf.namelist()
            for f in contents:
                try:
                    zipf.extract(f, extract_path)
                except IOError:
                    logger.warning(f"Failed to extract zipfile {f}."
                                        " Possible duplicate, removing the old file")
                    os.remove(path.join(extract_path, f))
                    zipf.extract(f, extract_path)

    def _cleanpath(self, i_path):
        abspath = path.abspath(expanduser(i_path))
        if os.path.isfile(abspath):
          return abspath
        return abspath + "/"

    def _makedirs(self, dir_path):
        dir_path = self._cleanpath(dir_path)
        if os.path.isfile(dir_path):
            dir_path = os.path.dirname(dir_path)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, mode=0o777, exist_ok=True)
            except OSError:
                logger.error(f"Failed to create directory {dir_path}")
                return None
        return dir_path

    def _download_file(self, url, headers, import_path, unpack = False):
        res = self._makedirs(import_path)
        if not res:
            return False

        filename = url.split('/')[-1]
        filepath = path.join(import_path, filename)
        with self._session.get(url, headers=headers, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.RequestException as err:
                logger.error(f"File download failed {filepath}")
                logger.debug(f"Request failed due to {err}")
                return False

            with open(filepath, 'wb') as f:
                logger.debug(f"Downloading file {filepath}")
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        if unpack:
            if filepath.endswith(".tar.gz"):
                self._extract_tar_file(filepath, import_path)
            elif filepath.endswith(".zip"):
                self._extract_zip_file(filepath, import_path)
            os.remove(filepath)


        return True

    def _download_zip(self, url, headers, import_path):
        res = self._makedirs(import_path)
        if not res:
            return False

        split_url = url.split('/')
        # This downloads a zip file of the contents
        zip_file = path.join(import_path, f'{split_url[-4]}_{split_url[-2]}.{split_url[-1]}')
        # Streaming enabled for large files...
        with self._session.get(url, headers=headers, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.RequestException as err:
                logger.error(f"File download failed {zip_file}")
                logger.debug(f"Request failed due to {err}")
                return False

            with open(zip_file, 'wb') as f:
                logger.debug(f"Downloading zip file {zip_file}")
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        with ZipFile(zip_file) as zipf:
            logger.debug(f"Extracting zip file {zip_file}")
            contents = zipf.namelist()
            gzf = [file for file in contents if file.endswith("tar.gz")]

            for gz in gzf:
                zipf.extract(gz, import_path)
                gzpath = path.join(import_path, gz)
                self._extract_tar_file(gzpath, import_path)
                os.remove(gzpath)

        os.remove(zip_file)
        return True

    def get_latest_version(self, ext_name):
        versions = self.get_extension_versions(ext_name)
        if not versions:
            logger.error(f"Failed to download extension metadata for {ext_name}")
            return False

        base_versions = [ver for ver in versions if ver.find("-") == -1]
        max_ver = base_versions[0]
        for ver in base_versions:
            max_ver = ver if version.parse(ver) > version.parse(max_ver) else max_ver

        return max_ver

    def pull_ext_metadata(self, ext_name, import_path, base_version):
        """ Downloads the extension metadata/interface yaml file from the base version

        Args:
            ext_name (str): extension name
            import_path (str): path to download the file
            base_version (str): base version string

        Returns:
            bool: download status
        """
        ext_url  = self._get_ext_file_download_url(ext_name, base_version, "extension.yaml")
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path)

    def pull_py_srcs(self, ext_name, import_path, base_version):
        """ Downloads the extension python source files from the base version

        Args:
            ext_name (str): extension name
            import_path (str): path to download the file
            base_version (str): base version string

        Returns:
            bool: download status
        """
        ext_url  = self._get_ext_file_download_url(ext_name, base_version, "py_srcs.tar.gz")
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path, True)

    def pull_ext_interface(self, ext_name, import_path, base_version):
        """ Downloads the interface tarball package for an extension

        Args:
            ext_name (str): extension name
            import_path (str): path to download the tarball
            base_version (str): extension base version

        Returns:
            bool: download status
        """
        versions = self.get_extension_versions(ext_name)
        if not versions:
            logger.error(f"Failed to download extension metadata for {ext_name}")
            return False

        base_versions = [ver for ver in versions if ver.find("-") == -1]
        if base_version not in base_versions:
            logger.error(f"Selected version is invalid")
            return False

        ext_url = self._get_ext_download_url(ext_name, base_version)
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_zip(ext_url, headers, import_path)

    def pull_ext_variant(self, ext_name, eid, import_path, ext_ver,
                       target_cfg: TargetConfig):
        """ Downloads the variant tarball package from an ngc repository
            *Contents are not extracted

        Args:
            ext_name (str): extension name
            eid (str): extension uuid
            import_path (str): path to download the tarball
            ext_ver (str): extension version
            target_cfg (TargetConfig): target config

        Returns:
            bool: success/failure
        """
        versions = self.get_extension_versions(ext_name)
        if not versions:
            logger.error(f"Failed to download extension metadata for {ext_name}")
            return False

        available_variants = [ver for ver in versions if ext_ver in ver and ver.find("-") != -1]
        variant = self._get_variant_ver_str(ext_ver, target_cfg)
        if variant not in available_variants:
            logger.error(f"Extension variant {variant} not found in NGC")
            logger.error(f"Available variants {available_variants}")
            return False

        filename = eid + ".tar.gz"
        ext_url = self._get_ext_file_download_url(ext_name, variant, filename)
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path)

    def pull_ext_target(self, ext_name, import_path, ext_ver):
        """ Downloads the target yaml file from the variant version of an extension

        Args:
            ext_name (str): extension name
            import_path (str): path to download the target yaml
            ext_ver (str): variant version of the extension

        Returns:
            bool: download status
        """
        ext_url  = self._get_ext_file_download_url(ext_name, ext_ver, "target.yaml")
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path)

    def _log_error_response(self, response):
        if response.text:
            json_loaded = json.loads(response.text)
            if json_loaded:
                logger.debug(json.dumps(json_loaded, indent=4))

class NGCClient:
    """Implementation of an NGC Client which interacts with NGC registry
    """

    def __init__(self, apikey, org, team = NGC_NO_TEAM):
        self._org = org
        self._team = team
        self._apikey = apikey
        self._v2_url = "https://api.ngc.nvidia.com/v2"
        self._auth_token = None
        self._auth_last_gen_time = None
        self._session = requests.session()
        retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[ 500, 502, 503, 504 ])
        self._session.mount("https://", HTTPAdapter(max_retries=retries,pool_connections=cpu_count(), pool_maxsize=cpu_count()))

        if org == NGC_NO_ORG:
            raise ValueError(f"ngc org must be configured to use ngc, cannot use {org}")

        if team == NGC_NO_TEAM:
            self._auth_url = f"https://authn.nvidia.com/token?service=ngc&scope=group/ngc:{org}"
        else:
            self._auth_url = f"https://authn.nvidia.com/token?service=ngc&scope=group/ngc:{org}&scope=group/ngc:{org}/{team}"

    @property
    def ngc_apikey(self):
        return self._apikey

    @ngc_apikey.setter
    def ngc_apikey(self, value):
        self._apikey = value

    @ngc_apikey.deleter
    def ngc_apikey(self):
        self._apikey = NGC_NO_API_KEY

    @property
    def ngc_org(self):
        return self._org

    @ngc_org.setter
    def ngc_org(self, value):
        if value is None:
            self._org = NGC_NO_ORG
        else:
            self._org = value

    @ngc_org.deleter
    def ngc_org(self):
        self._org = NGC_NO_ORG

    @property
    def ngc_team(self):
        return self._team

    @ngc_team.setter
    def ngc_team(self, value):
        if value is None:
            self._team = NGC_NO_TEAM
        else:
            self._team = value

    @ngc_team.deleter
    def ngc_team(self):
        self._team = NGC_NO_TEAM

    def _config_api(self):
        if self._org == NGC_NO_ORG:
            logger.warning("ngc org name is not configured")
            return None

        if self._team == NGC_NO_TEAM:
            return f"/org/{self._org}/"

        return f"/org/{self._org}/team/{self._team}/"

    def _get_ngc_auth_token(self):
        """ Generate a new auth token
        """

        # Check if auth token has expired
        if self._auth_token and (datetime.datetime.now() -
                                 self._auth_last_gen_time).total_seconds() < NGC_AUTH_TOKEN_TIMEOUT:
            logger.debug(f"Reusing auth token {datetime.datetime.now()}")
            return self._auth_token

        logger.debug(f"Fetching new Auth Token {datetime.datetime.now()}")

        # Generate authorization based on your API Key (standard base64 encoding)
        auth_string = f'$oauthtoken:{self._apikey}'
        encoded_bytes = base64.b64encode(auth_string.encode('utf-8'))
        encoded_auth_str = str(encoded_bytes, 'utf-8')

        # Request Headers
        headers = {
            'Authorization': f'Basic {encoded_auth_str}',
            'Accept': '*/*',
            'Cache-Control': 'no-cache',
            'Host': 'authn.nvidia.com',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'cache-control': 'no-cache'
        }

        # Execute request
        response = self._session.get(self._auth_url, headers=headers)

        # verify API key is valid
        if response.status_code != 200:
            logger.error("Invalid NGC API Key")
            self._log_error_response(response)
            return None

        # Convert response to JSON
        json_response = json.loads(response.text)
        auth_token = json_response['token']
        self._auth_last_gen_time = datetime.datetime.now()
        self._auth_token = auth_token
        return auth_token

    def _get_default_ext_payload(self):
        return {
                'application': 'OTHER',
                'builtBy': '{AUTHOR}',
                'shortDescription': '{UUID}', # To be updated when we have
                'description': '{DESCRIPTION}',
                'displayName': '{EXTENSION_NAME}',
                'trainingFramework': 'Other',
                'modelFormat': 'gxf',
                'labels': {'LABELS'},
                'name': '{EXTENSION_NAME}',
                'precision': 'OTHER',
                'publisher': "{AUTHOR}",
                'publicDatasetUsed': {
                    'license': 'None',
                    'link': 'None',
                    'name': 'None'
                }
        }

    def _get_ext_create_url(self):
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + "resources"

    def _get_ext_remove_url(self, ext_name):
        config = self._config_api()
        if not config:
            return None

        ext_name = ext_name.lower()
        return self._v2_url + config + f"resources/{ext_name}"

    def _get_ext_remove_version_url(self, ext_name, version):
        config = self._config_api()
        if not config:
            return None

        ext_name = ext_name.lower()
        version = version.lower()
        return self._v2_url + config + f"resources/{ext_name}/versions/{version}"

    def _get_default_version_payload(self):
        return {
                'id': 1,
                'ownerName': '{AUTHOR}',
                'versionId': '{VERSION}'
        }

    def _get_ext_version_url(self, ext_name):
        ext_name = ext_name.lower()
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + f"resources/{ext_name}/versions"

    def _get_ext_file_upload_url(self, ext_name, version, file_name):
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + f"resources/{ext_name}/versions/{version}/files/{file_name}"

    def _get_ext_upload_finish_url(self, ext_name, version):
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + f"resources/{ext_name}/versions/{version}"

    def _get_ext_list_url(self):
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + f"resources/"

    def _get_ext_download_url(self, ext_name, ext_version):
        ext_name = ext_name.lower()
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + f"resources/{ext_name}/versions/{ext_version}/zip"

    def _get_ext_file_download_url(self, ext_name, ext_version, filename):
        ext_name = ext_name.lower()
        filename = filename.lower()
        config = self._config_api()
        if not config:
            return None

        return self._v2_url + config + f"resources/{ext_name}/versions/{ext_version}/files/{filename}"

    def _get_request_headers(self):
        auth_token = self._get_ngc_auth_token()
        if auth_token is None:
            return None

        headers = {
            'Content-Type': "application/json",
            'Accept': "application/json",
            'Authorization': f'Bearer {auth_token}',
            'Cache-Control': "no-cache",
            'Host': "api.ngc.nvidia.com",
            'Accept-Encoding': "gzip, deflate",
            'Connection': "keep-alive",
            'cache-control': "no-cache"
        }
        return headers

    def _get_download_headers(self, ext_url):
        auth_token = self._get_ngc_auth_token()
        if auth_token is None:
            return None

        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Accept': '*/*',
            'Cache-Control': 'no-cache',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': f"{ext_url}",
            'Connection': 'keep-alive',
            'cache-control': 'no-cache'
        }
        return headers

    def _get_base_ver_str(self, ext: Extension):
        return ext.version.replace(" ", "_").lower()

    def _get_variant_ver_str(self, version, target_cfg: TargetConfig):
        distro = target_cfg.platform.distribution
        arch = target_cfg.platform.arch
        opsys = target_cfg.platform.os

        compute_str = ""
        compute_str  += f"-cuda-{target_cfg.compute.cuda}" if target_cfg.compute.cuda else ""
        compute_str  += f"-cudnn-{target_cfg.compute.cudnn}" if target_cfg.compute.cudnn else ""
        compute_str  += f"-trt-{target_cfg.compute.tensorrt}" if target_cfg.compute.tensorrt else ""
        compute_str  += f"-ds-{target_cfg.compute.deepstream}" if target_cfg.compute.deepstream else ""
        compute_str  += f"-triton-{target_cfg.compute.triton}" if target_cfg.compute.triton else ""
        compute_str  += f"-vpi-{target_cfg.compute.vpi}" if target_cfg.compute.vpi else ""

        version_str = f"{version}-{opsys}-{arch}-{distro}{compute_str}"
        return version_str.replace(" ", "_").lower()

    def create_base_version(self, ext: Extension):
        logger.debug("Creating base version")
        version_payload = self._get_default_version_payload()
        version_payload["ownerName"] = ext.author
        version_payload["versionId"] = self._get_base_ver_str(ext)
        version_payload = json.dumps(version_payload, indent=4)

        version_url = self._get_ext_version_url(ext.name.lower())
        if version_url is None:
            logger.warning("Failed to create version url")
            return False

        headers = self._get_request_headers()
        if headers is None:
            return False

        response = self._session.post(version_url, data=version_payload, headers=headers)
        if response.status_code == 409:
            logger.error(f"Extension interface already exists in NGC : {self._get_base_ver_str(ext)}")
            self._log_error_response(response)
            return False

        if response.status_code != 200:
            logger.error(f"Base version create request failed for extension {ext.name}")
            logger.debug(f"URL: {version_url}, Status Code: {response.status_code}")
            self._log_error_response(response)
            return False

        logger.debug("Base Version created successfully")
        return True


    def create_variant_version(self, ext: Extension,
                                 target_cfg: TargetConfig,
                                 repo_path):
        logger.debug("Creating variant version")
        variant_ver_str = self._get_variant_ver_str(ext.version, target_cfg)

        version_payload = self._get_default_version_payload()
        version_payload["ownerName"] = ext.author
        version_payload["versionId"] = variant_ver_str
        version_payload = json.dumps(version_payload, indent=4)

        version_url = self._get_ext_version_url(ext.name.lower())
        if version_url is None:
            logger.warning("Failed to create version url")
            return False

        headers = self._get_request_headers()
        if headers is None:
            return False

        response = self._session.post(version_url, data=version_payload, headers=headers)
        if response.status_code == 409:
            logger.error(f"Variant already exists in NGC : {variant_ver_str}")
            self._log_error_response(response)
            return False

        if response.status_code != 200:
            logger.error(f"Variant version create request failed for extension {ext.name}")
            logger.debug(f"URL: {version_url}, Status Code: {response.status_code}")
            self._log_error_response(response)
            return False

        logger.debug("Variant version created successfully")
        return True

    def _upload_file(self, filename, filepath, url):
        auth_token = self._get_ngc_auth_token()
        if auth_token is None:
            return False

        headers = {
            'Accept': "application/json",
            'Expect': "100-continue",
            'Authorization': f"Bearer {auth_token}",
            'Cache-Control': "no-cache",
            'Host': "api.ngc.nvidia.com",
            'Accept-Encoding': "gzip, deflate",
            'Connection': "keep-alive"
        }

        if not path.exists(filepath):
            logger.warning(f"missing file for upload {filepath}")
            return False

        if url is None:
            logger.warning("invalid upload url")
            return False

        files = {
            'file': (filename, open(filepath, "rb"))
        }
        response = self._session.put(url, files = files, headers = headers)
        if response.status_code != 200:
            logger.error(f"File upload request failed for {filename}")
            logger.debug(f"URL: {url}, Status Code: {response.status_code}")
            self._log_error_response(response)
            return False

        return True

    def _upload_archive(self, version_str, ext: Extension, pkg: Packager):
        ext_url = self._get_ext_file_upload_url(ext.name.lower(), version_str, pkg.package_name)
        result = self._upload_file(pkg.package_name, pkg.package_path, ext_url)
        if not result:
            logger.error(f"Extension files upload failed for {ext.name}")
            return False

        return True

    def finish_file_upload(self, url):
        logger.debug("Finishing file upload")
        upload_complete_payload= {
            "status": "UPLOAD_COMPLETE",
          }
        upload_complete_payload = json.dumps(upload_complete_payload, indent=4)
        headers = self._get_request_headers()
        if headers is None:
            return False

        response = self._session.patch(url, data=upload_complete_payload, headers=headers)

        if response.status_code != 200:
            logger.error("Upload finish request failed")
            logger.debug(f"URL: {url}, Status Code: {response.status_code}")
            self._log_error_response(response)
            return False

        return True

    def upload_ext_interface_files(self, ext: Extension, repo_path):
        logger.debug("Uploading interface files ")
        version_str = self._get_base_ver_str(ext)
        metadata_name = "extension.yaml"
        metadata_path = path.join(repo_path, ext.uuid, metadata_name)
        metadata_url = self._get_ext_file_upload_url(ext.name.lower(), version_str, metadata_name)

        result = self._upload_file(metadata_name, metadata_path, metadata_url)
        if not result:
            logger.error(f"Metadata upload failed for extension {ext.name}")
            return False

        # LICENSE file is currently optional, upload if present
        license_name = "LICENSE"
        license_path = path.join(repo_path, ext.uuid, license_name)
        if path.exists(license_path):
            license_url = self._get_ext_file_upload_url(ext.name.lower(), version_str, license_name)
            if not self._upload_file(license_name, license_path, license_url):
                logger.error(f"License upload failed for extension {ext.name}")
                return False

        if ext.headers:
            pkg = Packager("./", "headers")
            headers_dir = path.join(repo_path, ext.uuid, "headers")
            result = pkg.addDirectory(headers_dir)
            if not result:
                return False
            pkg.zip()

            if not self._upload_archive(version_str, ext, pkg):
                return False

            try:
                os.remove(pkg.package_path)
            except IOError:
                logger.debug(f"Failed to clean package {pkg.package_path}")

        if ext.python_sources:
            pkg = Packager("./", "py_srcs")
            py_srcs_dir = path.join(repo_path, ext.uuid, "py_srcs")
            result = pkg.addDirectory(py_srcs_dir)
            if not result:
                return False
            pkg.zip()

            if not self._upload_archive(version_str, ext, pkg):
                return False

            try:
                os.remove(pkg.package_path)
            except IOError:
                logger.debug(f"Failed to clean package {pkg.package_path}")

        upload_finish_url = self._get_ext_upload_finish_url(ext.name.lower(), version_str)
        result = self.finish_file_upload(upload_finish_url)
        if not result:
            return False

        return True

    def upload_extension_variant(self, ext: Extension,
                                     target_cfg: TargetConfig,
                                     repo_path):
        logger.debug("Uploading files ")

        pkg = Packager("./", ext.uuid)
        variant_path = path.join(repo_path, get_ext_subdir(ext.uuid, ext.version, target_cfg))
        result = pkg.addDirectory(variant_path, exclude_files=["manifest.yaml","target.yaml"])
        if not result:
            return False
        pkg.zip()

        target_name = "target.yaml"
        target_path = path.join(variant_path, target_name)
        target_yaml = YamlLoader().load_yaml(target_path)
        if not target_yaml:
            return False

        # Update the package hash
        target_yaml["sha256"] = pkg.sha256_hash()
        with open(target_path, "w+") as f:
            yaml.dump(target_yaml, f, default_flow_style=False, sort_keys=False)

        version_str = self._get_variant_ver_str(ext.version, target_cfg)

        if not self._upload_archive(version_str, ext, pkg):
            return False

        try:
            os.remove(pkg.package_path)
        except IOError:
            logger.debug(f"Failed to clean package {pkg.package_path}")

        target_url = self._get_ext_file_upload_url(ext.name.lower(), version_str, target_name)
        result = self._upload_file(target_name, target_path, target_url)
        if not result:
            logger.error(f"Target file upload failed for extension {ext.name}")
            return False

        upload_finish_url = self._get_ext_upload_finish_url(ext.name.lower(), version_str)
        result = self.finish_file_upload(upload_finish_url)
        if not result:
            return False

        logger.debug("Files uploaded successfully")
        return True

    def create_extension(self, ext: Extension, force = False):
        logger.debug("Creating extensions: " + ext.name)
        ext_payload = self._get_default_ext_payload()
        ext_payload["builtBy"] = ext.author
        ext_payload["publisher"] = ext.author
        ext_payload["description"] = ext.description
        # ext_payload["shortDescription"] = ext.description
        # Using uuid in short desc until gxf has formal support from ngc
        ext_payload["shortDescription"] = ext.uuid
        ext_payload["displayName"] = ext.name
        ext_payload["name"] = ext.name.lower()
        ext_payload["labels"] = ext.labels
        ext_payload = json.dumps(ext_payload, indent=4)

        create_url = self._get_ext_create_url()
        if create_url is None:
            logger.warning("failed to create extension url")
            return False

        headers = self._get_request_headers()
        if headers is None:
            return False

        response = self._session.post(create_url, data=ext_payload, headers=headers)
        if response.status_code == 409:
            logger.debug(f"Extension already exists in NGC : {ext.name}")
            self._log_error_response(response)

            # Override extensions by first removing the current interface and variants
            if force:
                logger.info(f"Extension override requested. Removing {ext.name}:{ext.version} ...")
                ext_versions = self.get_extension_versions(ext.name)
                if not ext_versions:
                    return False

                ext_variants = [ver for ver in ext_versions if ext.version == ver[: ver.find("-")]]
                ext_versions = ext_variants + [str(ext.version)]

                for v in ext_versions:
                    if not self._remove_ext_version(ext.name, v):
                        logger.error(f"Failed to remove extension {ext.name}:{v} while trying to overwrite it")
                        return False

                logger.debug(f"Extension removed from NGC {ext.name}:{ext.version}")
            return True

        if response.status_code != 200:
            logger.error(f"Create extension request failed : {ext.name}")
            logger.debug(f"URL: {create_url}, Status Code: {response.status_code}")
            self._log_error_response(response)
            return False

        logger.debug("Extension created successfully")
        return True

    def get_extension_list(self):
        list_url = self._get_ext_list_url()
        if list_url is None:
            logger.warning("failed to create extension list url")
            return None

        headers = self._get_request_headers()
        if headers is None:
            return None

        response = self._session.get(list_url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Extension list request failed for org {self.ngc_org}")
            logger.debug(f"URL: {list_url}, Status Code: {response.status_code}")
            return None

        payload = json.loads(response.text)
        if "recipes" not in payload:
            logger.warning(f"No extensions found in org: {self._org} and team: {self._team}")
            return {}

        exts = {}
        for recipe in payload["recipes"]:
            if recipe["modelFormat"] == "gxf":
                uuid = recipe["shortDescription"]
                if uuid_validator(uuid):
                    exts[uuid] = recipe["name"]

        logger.debug("Extension list fetched successfully")
        return exts

    def publish_extension_interface(self, ext: Extension, repo_path, force=False):
        result = self.create_extension(ext, force)
        if not result:
            return False

        result = self.create_base_version(ext)
        if not result:
            return False

        result = self.upload_ext_interface_files(ext, repo_path)
        if not result:
            return False

        return True

    def publish_extension_variant(self, ext: Extension,
                                        target_cfg: TargetConfig,
                                        repo_path):
        versions = self.get_extension_versions(ext.name)
        if not versions:
            return False

        if ext.version not in versions:
            logger.error(f"Extension interface not found for {ext.name}")
            return False

        result = self.create_variant_version(ext, target_cfg, repo_path)
        if not result:
            return False

        result = self.upload_extension_variant(ext, target_cfg, repo_path)
        if not result:
            return False

        return True

    def remove_extension_interface(self, ext_name, version):
        if not isinstance(ext_name, str):
            return False

        ext_versions = self.get_extension_versions(ext_name)
        if not ext_versions:
            return False

        if version not in ext_versions:
            logger.error(f"Extension version not found in NGC : {version}")
            return False

        ext_variants = [ver for ver in ext_versions if version == ver[: ver.find("-")]]
        if ext_variants:
            logger.error("Extension variants need to be removed before"
                               f" the extension interface {ext_variants}")
            return False

        remove_url = None
        # If extension interface is the only version present, remove the resource
        # else remove just the interface version
        if len(ext_versions) == 1:
            remove_url = self._get_ext_remove_url(ext_name)
        else:
            remove_url = self._get_ext_remove_version_url(ext_name, version)
        if not remove_url:
            return False

        headers = self._get_request_headers()
        if headers is None:
            return False

        response = self._session.delete(remove_url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to remove extension {ext_name}")
            self._log_error_response(response)
            return False

        logger.debug(f"Removed extension {ext_name} from NGC successfully")
        return True

    def remove_extension_variant(self, ext_name, uuid, version, target_cfg: TargetConfig):
        ext_versions = self.get_extension_versions(ext_name)
        if not ext_versions:
            return False

        variant_ver_str = self._get_variant_ver_str(version, target_cfg)
        if variant_ver_str not in ext_versions:
            logger.error(f"Extension variant not found in NGC {ext_name}:{variant_ver_str}")
            return False

        return self._remove_ext_version(ext_name, variant_ver_str)

    def _remove_ext_version(self, ext_name, ext_version):
        remove_url = self._get_ext_remove_version_url(ext_name, ext_version)
        if not remove_url:
            return False

        headers = self._get_request_headers()
        if headers is None:
            return False

        response = self._session.delete(remove_url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to remove extension variant {ext_name}:{ext_version}")
            self._log_error_response(response)
            return False

        return True

    def _extract_tar_file(self, filepath, extract_path):
        with TarFile.open(filepath, mode="r:gz") as tarball:
            logger.debug(f"Extracting file {filepath} to {extract_path}")
            for tarf in tarball:
                try:
                    tarball.extract(tarf.name, extract_path)
                except IOError:
                    logger.warning(f"Failed to extract tarfile {tarf.name}."
                                        " Possible duplicate, removing the old file")
                    os.remove(path.join(extract_path, tarf.name))
                    tarball.extract(tarf.name, extract_path)

    def _extract_zip_file(self, filepath, extract_path):
        with ZipFile(filepath) as zipf:
            logger.debug(f"Extracting zip file {filepath} to {extract_path}")
            contents = zipf.namelist()
            for f in contents:
                try:
                    zipf.extract(f, extract_path)
                except IOError:
                    logger.warning(f"Failed to extract zipfile {f}."
                                        " Possible duplicate, removing the old file")
                    os.remove(path.join(extract_path, f))
                    zipf.extract(f, extract_path)

    def _cleanpath(self, i_path):
        abspath = path.abspath(expanduser(i_path))
        if os.path.isfile(abspath):
          return abspath
        return abspath + "/"

    def _makedirs(self, dir_path):
        dir_path = self._cleanpath(dir_path)
        if os.path.isfile(dir_path):
            dir_path = os.path.dirname(dir_path)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, mode=0o777, exist_ok=True)
            except OSError:
                logger.error(f"Failed to create directory {dir_path}")
                return None
        return dir_path

    def get_extension_versions(self, ext_name):
        ext_name = ext_name.lower()

        version_url = self._get_ext_version_url(ext_name)
        if version_url is None:
            logger.warning("Failed to create version url")
            return None

        headers = self._get_request_headers()
        if headers is None:
            return None

        response = self._session.get(version_url, headers=headers)

        if response.status_code != 200:
            logger.error(f"Extension not found in NGC : {ext_name}")
            self._log_error_response(response)
            return None

        payload = json.loads(response.text)
        if "recipeVersions" not in payload:
            logger.warning(f"No extension versions found for extension {ext_name}")
            return None

        versions = [ver["versionId"] for ver in payload["recipeVersions"]]
        return versions

    def _download_file(self, url, headers, import_path, unpack = False):
        res = self._makedirs(import_path)
        if not res:
            return False

        filename = url.split('/')[-1]
        filepath = path.join(import_path, filename)
        with self._session.get(url, headers=headers, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.RequestException as err:
                logger.error(f"File download failed {filepath}")
                logger.debug(f"Request failed due to {err}")
                return False

            with open(filepath, 'wb') as f:
                logger.debug(f"Downloading file {filepath}")
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        if unpack:
            if filepath.endswith(".tar.gz"):
                self._extract_tar_file(filepath, import_path)
            elif filepath.endswith(".zip"):
                self._extract_zip_file(filepath, import_path)
            os.remove(filepath)

        return True


    def _download_zip(self, url, headers, import_path):
        res = self._makedirs(import_path)
        if not res:
            return False

        split_url = url.split('/')
        # This downloads a zip file of the contents
        zip_file = path.join(import_path, f'{split_url[-4]}_{split_url[-2]}.{split_url[-1]}')
        # Streaming enabled for large files...
        with self._session.get(url, headers=headers, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.RequestException as err:
                logger.error(f"File download failed {zip_file}")
                logger.debug(f"Request failed due to {err}")
                return False

            with open(zip_file, 'wb') as f:
                logger.debug(f"Downloading zip file {zip_file}")
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        with ZipFile(zip_file) as zipf:
            logger.debug(f"Extracting zip file {zip_file}")
            contents = zipf.namelist()
            gzf = [file for file in contents if file.endswith("tar.gz")]

            for gz in gzf:
                zipf.extract(gz, import_path)
                gzpath = path.join(import_path, gz)
                self._extract_tar_file(gzpath, import_path)
                os.remove(gzpath)

        os.remove(zip_file)
        return True

    def get_latest_version(self, ext_name):
        versions = self.get_extension_versions(ext_name)
        if not versions:
            logger.error(f"Failed to download extension metadata for {ext_name}")
            return False

        base_versions = [ver for ver in versions if ver.find("-") == -1]
        max_ver = base_versions[0]
        for ver in base_versions:
            max_ver = ver if version.parse(ver) > version.parse(max_ver) else max_ver

        return max_ver

    def pull_ext_metadata(self, ext_name, import_path, base_version):
        """ Downloads the extension metadata/interface yaml file from the base version

        Args:
            ext_name (str): extension name
            import_path (str): path to download the file
            base_version (str): base version string

        Returns:
            bool: download status
        """
        ext_url  = self._get_ext_file_download_url(ext_name, base_version, "extension.yaml")
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path)

    def pull_py_srcs(self, ext_name, import_path, base_version):
        """ Downloads the extension python source files from the base version

        Args:
            ext_name (str): extension name
            import_path (str): path to download the file
            base_version (str): base version string

        Returns:
            bool: download status
        """
        ext_url  = self._get_ext_file_download_url(ext_name, base_version, "py_srcs.tar.gz")
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path, True)

    def pull_ext_interface(self, ext_name, import_path, base_version):
        """ Downloads the interface tarball package for an extension

        Args:
            ext_name (str): extension name
            import_path (str): path to download the tarball
            base_version (str): extension base version

        Returns:
            bool: download status
        """
        versions = self.get_extension_versions(ext_name)
        if not versions:
            logger.error(f"Failed to download extension metadata"
                               f" for {ext_name}")
            return False

        base_versions = [ver for ver in versions if ver.find("-") == -1]
        if base_version not in base_versions:
            logger.error(f"Selected version is invalid")
            return False

        ext_url = self._get_ext_download_url(ext_name, base_version)
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_zip(ext_url, headers, import_path)

    def pull_ext_variant(self, ext_name, eid, import_path, ext_ver,
                        target_cfg: TargetConfig):
        """ Downloads the variant tarball package from an ngc repository
            *Contents are not extracted

        Args:
            ext_name (str): extension name
            eid (str): extension uuid
            import_path (str): path to download the tarball
            ext_ver (str): extension version
            target_cfg (TargetConfig): target config

        Returns:
            bool: download status
        """
        versions = self.get_extension_versions(ext_name)
        if not versions:
            logger.error(f"Failed to download extension metadata for {ext_name}")
            return False

        available_variants = [ver for ver in versions if ext_ver in ver and ver.find("-") != -1]
        variant = self._get_variant_ver_str(ext_ver, target_cfg)
        if variant not in available_variants:
            logger.error(f"Extension variant {variant} not found in NGC")
            logger.error(f"Available variants {available_variants}")
            return False

        filename = eid + ".tar.gz"
        ext_url = self._get_ext_file_download_url(ext_name, variant, filename)
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        if not self._download_file(ext_url, headers, import_path):
            logger.error(f"Failed to download variant {variant}")
            return False

        return True

    def pull_ext_target(self, ext_name, import_path, ext_ver):
        """ Downloads the target yaml file from the variant version of an extension

        Args:
            ext_name (str): extension name
            import_path (str): path to download the target yaml
            ext_ver (str): variant version of the extension

        Returns:
            bool: download status
        """
        ext_url  = self._get_ext_file_download_url(ext_name, ext_ver, "target.yaml")
        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        headers = self._get_download_headers(ext_url)
        if headers is None:
            return False

        return self._download_file(ext_url, headers, import_path)

    def _log_error_response(self, response):
        if response.text:
            try:
                json_loaded = json.loads(response.text)
                if json_loaded:
                    logger.debug(json.dumps(json_loaded, indent=4))
            except json.decoder.JSONDecodeError:
                  pass
