#!/bin/bash
#####################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# This script cleans the workspace, creates the test results in the desired xml format for Jenkins,
# and uploads the most recent version of the test results to the artifactory
set -e

# Cleanup to make sure we have the new testlogs since jenkins sometimes delays this
echo "Cleanup testlog folder"
rm -rf bazel_tests_directory

mkdir -p bazel_tests_directory

echo "Find the test reports in bazel directory"
# Find the test reports in bazel directory
BAZEL_XML_PATHS=$(find -L bazel-testlogs -name "*test.xml")

count=0
# Loop to copy the report into desired directory so Jenkins can see them.
echo "Loop to copy the report into desired directory"
for TEST in ${BAZEL_XML_PATHS}
do
    let "count+=1"
    mkdir bazel_tests_directory/$count
    mv -f ${TEST} bazel_tests_directory/$count/
done

echo "Find missing and existing classnames"
# Find missing and existing classnames
TEST_XMLS_WITH_CLASSNAME=$(grep -iRl "classname" bazel_tests_directory/)
TEST_XMLS_WITHOUT_CLASSNAME=$(grep -RL "classname" bazel_tests_directory/)

echo "Loops to format results so that Jenkins plugin can read them"
# Loops to format results so that Jenkins plugin can read them
for TEST_XML in ${TEST_XMLS_WITH_CLASSNAME}
do
    sed -i 's/classname="/&x86\./g' $TEST_XML
done

for TEST_XML in ${TEST_XMLS_WITHOUT_CLASSNAME}
do
    sed -i 's/<testcase /&classname="x86\.x86_NoClassFound" /g' $TEST_XML
done

echo "Package and upload to the artifactory"
# Package and upload to the artifactory
tar -cvf bazel_tests.tar bazel_tests_directory
curl -$1:$2 -T bazel_tests.tar $3
