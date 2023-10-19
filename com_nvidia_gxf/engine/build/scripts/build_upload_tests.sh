#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# This script builds all the tests for Jetson devices and uploads all binaries and
# dependencies to the artifactory.

set -e

# constants
BAZEL_BIN="bazel-bin"
LOCAL_DEVICE="x86_64"

# Helper functions to be used in this script.
# Prints the error message and exits the script.
error_and_exit() {
    printf "Error: $1\n"
    exit 1
}

# Prints the warning message.
warn() {
    printf "Warning: $1\n"
}

# used arguments with default values
UNAME=$USER
REMOTE_USER=nvidia #if no username provided
REMOTE_USER_SET=false
ENABLE_CXX17=false
BAZEL_CONFIGS=""

# get command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
    -d | --device)
        DEVICE="$2"
        ;;
    -h | --host)
        HOST="$2"
        ;;
    -u | --user)
        UNAME="$2"
        ;;
    -a | --artifactory)
        ARTIF="$2"
        ;;
    -p | --pword)
        PWORD="$2"
        ;;
    -r | --run)
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
    --cxx17)
        ENABLE_CXX17=true
        shift
        continue
        ;;
    *)
        error_and_exit "Invalid arguments: %1 %2\n"
        ;;
    esac
    shift
    shift
done

if [ -z "$DEVICE" ]; then
    error_and_exit "Desired target device must be specified with -d DEVICE. Valid choices: 'hp11_sbsa', 'jetpack51' 'x86_64'"
fi

if [ "$ENABLE_CXX17" = true ]; then
    BAZEL_CONFIGS=" --config=cxx17 "
fi

#create the necessary files & folders
>all_tests
>remote_tests
>dependencies_test
>dep_loc
>dep_loc_tmp
>exe_loc_tmp
>exe_tmp
>py_dep
mkdir test_dep || rm -rf test_dep/*
mkdir executables || rm -rf executables/*
mkdir bazel-bin-copy || rm -rf bazel-bin-copy/*

# find all the tests that are written for the given configuration, except for the ones tagged "host", "manual", "pytest", "performance"
bazel query 'attr(tags, "pytest", tests(...)) except attr(tags, "host|manual|performance", ...)'>>python_tests
bazel query 'tests(...) except attr(tags, "host|manual|pytest|performance", ...)'>>all_tests
# string manipulation to feed the tests to the bazel build
sed -i 's/\s.*$//' ./all_tests

# remove the tests that we do not want to run on jetson
sed -i '/cpplint/d' ./all_tests
sed -i '/gxflint/d' ./all_tests
sed -i '/_check_json/d' ./all_tests

tests_to_build=$(cat all_tests)

echo "================================================================================"
echo "Building for target platform '$DEVICE'"
echo "================================================================================"

# build everything for the given configuration except for the ones tagged "host", "manual", "pytest", "performance"
bazel build --config $DEVICE $BAZEL_CONFIGS --build_tag_filters=-host,-manual,-pytest,-performance $tests_to_build

# string formatting
sed -r -i 's/:/\//g' all_tests
cut -c 3- all_tests > all_tests_formatted

cat all_tests_formatted | while read line; do
    # paths to the test targets are stored in the $line variable now
    # find the dependencies for the tests and save them in a file to read later
    if [ -f "bazel-bin/$line" ]; then
        objdump -x -p bazel-bin/$line | grep NEEDED | cut -d "." -f 1,2,3 >>dependencies_test
    fi
done

# remove the duplicate dependencies
awk '!x[$0]++' dependencies_test > uniq_dep_tmp

# formatting
cut -c 24- uniq_dep_tmp > uniq_dep

###################################################################################################
### do the same steps for python tests, with the only difference of using aquery for dependencies
sed -i 's/\s.*$//' ./python_tests
sed -i '/cpplint/d' ./python_tests
sed -i '/gxflint/d' ./python_tests
sed -i '/_check_json/d' ./python_tests

pytests_to_build=$(cat python_tests)

echo "================================================================================"
echo " PYTHON TESTS BUILD"
echo "================================================================================"

bazel build --config $DEVICE $BAZEL_CONFIGS $pytests_to_build

cat python_tests | while read line; do
    # paths to the test targets are stored in the $line variable now
    # find the dependencies for the tests and save them in a file to read later
    bazel aquery $line --config $DEVICE --output=jsonproto | jq -r '
     .artifacts |
     .[] |
     .execPath' >> aquery_out
done

# convert the output in python tests from:
# "//engine/pyalice/tests:application_test" to
# "engine/pyalice/tests/application_test" and save
sed -r -i 's/:/\//g' python_tests
cut -c 3- python_tests > python_tests_formatted

# keep only .so files from aquery output
[ -f "./aquery_out" ] && sed -i '/\.so/!d' ./aquery_out

# remove the duplicate dependencies
[ -f "aquery_out" ] && awk '!x[$0]++' aquery_out > aquery_deps

# only keep the shared library name from the dep. path
[ -f "aquery_deps" ] && sed 's@.*/@@' aquery_deps > python_dependencies

[ -f "python_dependencies" ] && cat python_dependencies >> uniq_dep
awk '!x[$0]++' uniq_dep > all_dep

###################################################################################################
bazel_cache=$(bazel info output_base)

EXCLUDE_STR='-not -path "*x86_64*" -not -path "*.sym" '
case "$DEVICE" in

  "hp11_sbsa" | "hp20_sbsa" | "hp21ea_sbsa" | "jetpack51")
    EXCLUDE_STR+='-not -path "*qnx*"'
    ;;

  "jetpack45" | "jetpack46" | "jetpack45_mccoy")
    EXCLUDE_STR+='-not -path "*qnx*" -not -path "*buildroot*"'
    ;;

  "driveqnx602" | "drivelinux601")
    EXCLUDE_STR+='-not -path "*linux*"'
    ;;

  *)
esac

cat all_dep | while read line; do
    # paths to the test targets are stored in the $line variable now
    # the bazel cache location might differ, need to figure out where it is for
    # the docker image this script is being run on
    # Looking for a file by name in cache and src
    dep_path=$(eval "find $bazel_cache $PWD -type f -iname '$line' $EXCLUDE_STR -print -quit" | head -1)
    if [ -n "$dep_path" ]; then
      if [ -L "$dep_path" ]; then
        echo "SYMLINK"
        dep_path=$(readlink -f $dep_path)
      else
        echo "NO SYMLINK $dep_path"
      fi
    else
      # Looking in a system folder
      dep_path=$(find /usr/aarch64-linux-gnu/lib/ -name "$line" -print -quit | head -1)
      if [ ! -e "$dep_path" ]; then
        # If not found trying to look for a different version (filename)
        dep_path=$(eval "find $bazel_cache $PWD -type f -iname '$line*' $EXCLUDE_STR -print -quit" | head -1)
      fi
    fi

    # if a file is found, add to the location list
    if [[ -e $dep_path ]]; then
        echo $dep_path >>dep_loc_tmp
    else
        echo "$line NOT FOUND"
    fi
done

# remove the duplicate dependencies
awk '!x[$0]++' dep_loc_tmp > dep_loc

cat dep_loc | while read line; do
    # paths to the dependencies are stored in the $line variable now
    # copy all the dependency files into one directory
    rsync -a -L $line $PWD/test_dep
done

cat all_dep | while read line; do
  shared_lib_path=$(find $PWD/test_dep/ -name "$line*")
  if [[ -e $shared_lib_path ]]; then
      rsync -a -L $shared_lib_path $PWD/test_dep/$line
  fi
done

# since bazel-bin is a symlink itself, and we will not copy symlinks after we create a copy directory of bazel-bin where the build artifacts are stored
rsync -a $PWD/bazel-bin/ bazel-bin-copy/
rsync -a bazel-bin-copy/ .
rm -rf bazel-bin-copy

echo "================================================================================"
echo "Creating Tar Ball"
echo "================================================================================"

# copy to the target device, excluding the files & directories that are not needed for tests to run
rsync -a -l -W --no-compress\
    --exclude '*.lo' --exclude '*.a' --exclude '*.d' --exclude '*.o'\
    --exclude 'bazel-genfiles' --exclude 'bazel-development' --exclude 'bazel-bin' --exclude 'doc'\
    --exclude 'bazel-engine' --exclude 'bazel-engine-public' --exclude 'bazel-sdk' --exclude 'bazel-sdk-public'\
    --exclude 'bazel-out' --exclude 'bazel-testlogs' --exclude 'bazel-workspace' --exclude 'coverity'\
    --exclude '.git' $PWD/\
    $PWD/jetson_artifactory
tar -cf gxf_engine.tar jetson_artifactory

echo "================================================================================"
echo "Uploading All Tests to Artifactory"
echo "================================================================================"
curl -$UNAME:$PWORD -T gxf_engine.tar $ARTIF
