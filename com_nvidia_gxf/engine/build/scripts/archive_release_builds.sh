#!/usr/bin/env bash
#####################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Generate an API key for artifactory and export it to the env as ARTIFACTORY_API_KEY
# Update the BUILD_TYPE, BUILD_TAG and ARCHIVE_URL and run script to archive engineering release builds

set -e

# Download
NIGHTLY_URL="https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/nightly"
BRANCH="release-23.03"
BUILD_TYPE="internal"
BUILD_TAG="20230322_ef8576b3"

# Upload
ARCHIVE_URL="https://urm.nvidia.com/artifactory/sw-isaac-gxf-generic-local/release/engineering/23.03"

# API key for artifacotry needed for upload
if [[ -z "$ARTIFACTORY_API_KEY" ]]; then
  echo "set ARTIFACTORY_API_KEY env var to proceed"
  exit 1
fi

temp_directory=$(mktemp -d -t gxf-XXXXXXXXXX)
echo "Using temporary directory path:" $temp_directory

wget -r -A "*${BUILD_TAG}_${BUILD_TYPE}*" -R "index.html*" --show-progress --no-check-certificate -nd "${NIGHTLY_URL}/${BRANCH}"  --directory-prefix=$temp_directory

for file in "$temp_directory"/*
do
    filename=$(basename ${file})
    echo "Uploading $filename ..."
    curl -H "X-JFrog-Art-Api:${ARTIFACTORY_API_KEY}" -T $file "$ARCHIVE_URL/$filename"
done

echo "All files uploaded to $ARCHIVE_URL"
rm -rf $temp_directory
