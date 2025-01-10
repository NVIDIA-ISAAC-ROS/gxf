#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################################
# script settings
set -e  # fail on errors
set -o pipefail  # handle errors on pipes

# constants
REQUESTED_BAZEL_VERSION="6.0.0"
LOCAL_DEVICE="x86_64"
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_NONE='\033[0m'
COLOR_YELLOW='\033[0;33m'

# Helper functions to be used in this script.
# Prints the error message and exits the script.
error_and_exit() {
  printf "${COLOR_RED}Error: $1${COLOR_NONE}\n"
  printf "  see: $0 --help\n"
  exit 1
}
# Prints the warning message.
warn() {
  printf "${COLOR_YELLOW}Warning: $1${COLOR_NONE}\n"
}
# Prints help message.
print_help_message() {
  printf "Usage: $0 -p <package> -h <IP> -d <device> [options]\n"
  printf "  -d|--device:       Desired target device.\n"
  printf "  -h|--host:         Host IP address.\n"
  printf "  -p|--package:      Package to deploy, for example: //foo/bar:tar.\n"
  printf "  -r|--run:          Run on remote.\n"
  printf "  -s|--symbols:      Preserve symbols when building.\n"
  printf "  -u|--user:         Local username, defaults to ${USER}.\n"
  printf "  -b|--bazel:        Run script with either bazel or dazel.\n"
  printf "  --update:          Only deploy files that have changed.\n"
  printf "  --remote_user:     Username on target device.\n"
  printf "  --deploy_path:     Destination on target device.\n"
  printf "  --help:            Display this message.\n"
}
# reverses an array
# first argument is array to reverse, second is output array
function reverse_array() {
    declare -n arr="$1" rev="$2"
    for i in "${arr[@]}"
    do
        rev=("$i" "${rev[@]}")
    done
}
# this joins an array with a string, i.e.
#
#   join_by "/" ("foo" "bar")
#
# results in
#
#   foo/bar
function join_by {
    local IFS="$1"
    declare -n arr="$2"
    shift
    echo "${arr[*]}"
}
# this queries the output files for a given target from bazel, and returns them as a list
function get_outputs_for_target {
    local TARGET=${1}
    local CONFIGURATION=${2}

    # run aquery once and store the output
    local AQUERY_OUTPUT=$(${BAZEL} aquery --config "${CONFIGURATION}" "${TARGET}" --output=jsonproto --noinclude_commandline)

    # the outputs are listed as IDs in the actions, which are indices into the artifacts
    local OUTPUT_IDS=$(echo "$AQUERY_OUTPUT" | jq -r '.actions | .[] | .outputIds | .[]')

    local FRAGMENTS=()

    for OUTPUT_ID in $OUTPUT_IDS; do
        # each artifact contains an index into the path, which is the last path segment
        local FRAGMENT_ID=$(echo "$AQUERY_OUTPUT" | jq -r ".artifacts[] | select(.id == $OUTPUT_ID) | .pathFragmentId")

        # traverse up the fragments until we find the root
        while [[ "$FRAGMENT_ID" != "null" ]]; do
            # get the current path fragment
            FRAGMENT=$(echo "$AQUERY_OUTPUT" | jq -r ".pathFragments[] | select(.id == $FRAGMENT_ID)")
            FRAGMENT_PATH=$(echo "$FRAGMENT" | jq -r '.label')

            # get the parent ID of the path
            PARENT_ID=$(echo "$FRAGMENT" | jq -r '.parentId')

            # assemble paths
            FRAGMENTS+=("$FRAGMENT_PATH")

            # repeat with parent
            FRAGMENT_ID=${PARENT_ID}
        done

        # we now have all the parts of the path, but in reverse order
        reverse_array FRAGMENTS REVERSED
        join_by "/" REVERSED

        # reset the fragments for the next output
        FRAGMENTS=()
    done
}

# Save this script's directory in case it is being called from an external workspace
dir_this_script=$(dirname "$0")

# used arguments with default values
UNAME=$USER
REMOTE_USER=nvidia
REMOTE_USER_SET=false
CACHE_SERVER_ARG=""
BAZEL=auto
BAZEL_CONFIGS=""

# print help and exit if no command line arguments
if [ $# -eq 0 ]; then
  print_help_message
  exit 0
fi

# get command line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--package)
      PACKAGE="$2"
      ;;
    -d|--device)
      DEVICE="$2"
      ;;
    -c|--cache)
      CACHE_SERVER_ARG="--remote_cache=$2"
      ;;
    -h|--host)
      HOST="$2"
      ;;
    -u|--user)
      UNAME="$2"
      ;;
    -b|--bazel)
      BAZEL="$2"
      ;;
    -s|--symbols)
      NEED_SYMBOLS="True"
      shift
      continue
      ;;
    -r|--run)
      NEED_RUN="True"
      shift
      continue
      ;;
    --remote_user)
      REMOTE_USER="$2"
      REMOTE_USER_SET=true
      ;;
    --deploy_path)
      DEPLOY_PATH="$2"
      ;;
    --update)
      NEED_UPDATE="True"
      shift
      continue
      ;;
    --debug)
      ENABLE_DEBUG=true
      shift
      continue
      ;;
    --help)
      print_help_message
      exit 0
      ;;
    *)
      error_and_exit "Error: Invalid arguments: ${1} ${2}"
  esac
  shift
  shift
done

if [ -z "$PACKAGE" ]; then
  error_and_exit "Package must be specified with -p //foo/bar:tar."
fi
if [[ $PACKAGE != //* ]]; then
  error_and_exit "Package must start with //. For example: //foo/bar:tar."
fi

if [ -z "$HOST" ]; then
  error_and_exit "Host IP must be specified with -h IP."
fi

if [ -z "$DEVICE" ]; then
  error_and_exit "Desired target device must be specified with -d DEVICE. Valid choices: 'hp21ea_sbsa' 'hp21ga_sbsa' 'jetpack60' 'jetpack61' 'x86_64'"
fi

if [[ $BAZEL != bazel && $BAZEL != dazel && $BAZEL != auto ]]; then
  error_and_exit "Invalid bazel argument '${BAZEL}'. Valid choices: 'auto', 'bazel', 'dazel'."
fi

if [[ $BAZEL == auto ]]; then
  if [ $(which dazel) ]; then
    BAZEL=dazel
  else
    BAZEL=bazel
  fi
fi

if [[ $DEVICE == driveqnx602 ]]; then
  if [[ -z "$DEPLOY_PATH" ]]; then
    error_and_exit "For QNX platform, deploy path must be specified with --deploy_path."
  fi
fi

if [ "$ENABLE_DEBUG" = true ]; then
    BAZEL_CONFIGS=" --config=debug "
fi

# Check bazel version before building.
BAZEL_VERSION=$($BAZEL version | grep 'Build label' | sed 's/Build label: //')
if [[ $BAZEL_VERSION != $REQUESTED_BAZEL_VERSION ]]; then
  error_and_exit \
    "GXF requires bazel version $REQUESTED_BAZEL_VERSION. Please verify your bazel with 'bazel version' command."
fi

# Check if we need ssh to deploy. Potentially overwrite REMOTE_USER.
SSH_NEEDED=true
if [[ $HOST == "localhost" || $HOST == "127.0.0.1" ]]; then
  # Check username
  if [[ $REMOTE_USER_SET == false ]]; then
    # If user has not explicitly set REMOTE_USER, set it to $USER
    echo "No remote user is specified. Using '$USER' for local deployment."
    REMOTE_USER=$USER
  elif [[ $REMOTE_USER != $USER ]]; then
    warn "This is a local deployment, but remote user is explicitly specified as '$REMOTE_USER'"
  fi
  if [[ $REMOTE_USER == $USER ]]; then
    SSH_NEEDED=false
  fi
  if [[ $DEVICE != $LOCAL_DEVICE ]]; then
    warn "Deploying a '$DEVICE' package to localhost"
  fi
fi

# Split the target of the form //foo/bar:tar into "//foo/bar" and "tar"
targetSplitted=(${PACKAGE//:/ })
if [[ ${#targetSplitted[@]} != 2 ]]; then
  error_and_exit "Package '$PACKAGE' must have the form //foo/bar:tar"
fi
PREFIX=${targetSplitted[0]:2}
TARGET=${targetSplitted[1]}

TARPATH=$(get_outputs_for_target $PREFIX:$TARGET $DEVICE | egrep '(\.tar|\.tar\.gz)$')
TARFILE=$(basename ${TARPATH})

echo "================================================================================"
echo "Building Minidump tools"
echo "================================================================================"
if [[ $BAZEL == dazel ]]; then
  source $dir_this_script/prepare_minidump_tools.sh --use_dazel && wait
else
  source $dir_this_script/prepare_minidump_tools.sh && wait
fi

# build the bazel package
echo "================================================================================"
echo "Building //$PREFIX:$TARGET for target platform '$DEVICE'"
echo "================================================================================"
$BAZEL build $CACHE_SERVER_ARG --config $DEVICE $BAZEL_CONFIGS $PREFIX:$TARGET --strip=always || exit 1

# Print a message with the information we gathered so far
echo "================================================================================"
echo "Deploying //$PREFIX:$TARGET ($EX) to $REMOTE_USER@$HOST under name '$UNAME'"
echo "================================================================================"

# unpack the package in the local tmp folder
rm -f /tmp/$TARFILE
cp $TARPATH /tmp/
rm -rf /tmp/$TARGET
mkdir /tmp/$TARGET
tar -xf /tmp/$TARFILE -C /tmp/$TARGET

# copy libgxf_core.so if not present
lib_gxf_core_path=`find . -name "libgxf_core.so" -type f`
if [ ! -z "$lib_gxf_core_path" ]
then
  mkdir -p /tmp/$TARGET/gxf_$DEVICE/core/
  cp -n gxf_$DEVICE/core/libgxf_core.so /tmp/$TARGET/gxf_$DEVICE/core/
fi

# Deploy directory
if [ -z "$DEPLOY_PATH" ]
then
  DEPLOY_PATH="/home/$REMOTE_USER/deploy/$UNAME/"
fi

# sync the package folder to the remote
REMOTE_USER_AND_HOST="$REMOTE_USER@$HOST:"
# Special case: Don't ssh if not needed
if [[ $SSH_NEEDED == false ]]; then
  REMOTE_USER_AND_HOST=""
fi
# Special case: don't delete existing folder, don't update newer destination files.
if [[ ! -z $NEED_UPDATE ]]; then
  RSYNC_POLICY="--update"
else
  RSYNC_POLICY="--delete"
fi

if [ $DEVICE == "driveqnx602" ]
then
  # QNX based system does not support rsync
  scp -rv /tmp/$TARGET $REMOTE_USER_AND_HOST$DEPLOY_PATH
else
  rsync -avz $RSYNC_POLICY --checksum --rsync-path="mkdir -p $DEPLOY_PATH/ && rsync" \
      /tmp/$TARGET $REMOTE_USER_AND_HOST$DEPLOY_PATH
fi

status=$?
if [ $status != 0 ]; then
  error_and_exit "rsync failed with exit code $status"
fi

if [[ -z $NEED_SYMBOLS ]]; then
  echo "================================================================================"
  echo "To grab symbols pass -s/--symbols"
  echo "================================================================================"
else
  echo "================================================================================"
  echo "Grabbing symbols"
  echo "================================================================================"
  # Retain symbols in all binaries
  $BAZEL build $CACHE_SERVER_ARG --copt -g --config $DEVICE $BAZEL_CONFIGS $PREFIX:$TARGET --strip=never || exit 1

  # Unpack the package in the local tmp folder
  rm -f /tmp/$TARFILE
  cp $TARPATH /tmp/
  rm -rf /tmp/$TARGET
  mkdir /tmp/$TARGET
  tar -xf /tmp/$TARFILE -C /tmp/$TARGET

  EXECUTABLES=$(find /tmp/$TARGET -executable -type f)
  for exe in $EXECUTABLES
  do
    if [[ ! -z $(file "$exe" | sed '/ELF/!d') ]]; then
      $dir_this_script/process_syms.sh $exe
    fi
  done
  wait
fi
echo "================================================================================"
printf "${COLOR_GREEN}Deployed${COLOR_NONE}\n"
echo "================================================================================"

if [[ ! -z $NEED_RUN ]]; then
  echo "================================================================================"
  echo "Running on Remote"
  echo "================================================================================"
  # echo "cd $DEPLOY_PATH/$TARGET; ./$PREFIX/${TARGET::-4}"
  ssh -t $REMOTE_USER@$HOST "cd $DEPLOY_PATH/$TARGET; ./$PREFIX/${TARGET::-4}"
fi
