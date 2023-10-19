#!/bin/bash
#####################################################################################
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# This script pulls the necessary files for running the tests on Jetson from the
# Artifactory and the runs all the tests.
# After tests are done running it reports the results back.

function throwErrors()
{
    set -e
}

function ignoreErrors()
{
    set +e
}

throwErrors # failing a command if result is non-null

rm -rf jetson_artifactory

curl -$1:$2 -O $3
tar -xf $4
rm -rf $4

pushd jetson_artifactory

rm -rf jetson_testlog
mkdir jetson_testlog
rm -rf aborted_testlog
rm -rf aborted_tests.xml
rm -rf jetson-test-out.txt

BUILD_TESTLOG=${PWD}/jetson_testlog
WS_ARTIFACTORY=$PWD
export LD_LIBRARY_PATH=$PWD/test_dep
total_aborted=0

ignoreErrors # ignore failures of commands until further notice

function xml_string_manipulation ()
{
    sed -i "s/insert_test_suite/$2/g" $1
    sed -i "s/insert_test_case/$3/g" $1
    sed -i 's/\x1b\[[0-9;]*m//g' $1
    cp $1 $PWD/jetson_testlog
    rm -rf $1
}

#Load the error message template, in case a test gets aborted
error_template_1=$(cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="1" failures="1" disabled="0" errors="0" time="0" name="AllTests">
  <testsuite name="insert_test_suite" tests="1" failures="1" disabled="0" errors="0" time="0">
    <testcase name="insert_test_case" status="run" time="0" classname="insert_test_suite">
      <failure message="TEST ABORTED. SEE STACK TRACE FOR TERMINAL OUTPUT:" type=""><![CDATA[
EOF
)
error_template_2=$(cat <<EOF
]]></failure>
    </testcase>
  </testsuite>
</testsuites>
EOF
)
error_template_3=$(cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="1" failures="1" disabled="0" errors="0" time="0" name="AllTests">
  <testsuite name="insert_test_suite" tests="1" failures="1" disabled="0" errors="0" time="0">
    <testcase name="insert_test_case" status="run" time="0" classname="insert_test_suite">
      <failure message="NO TESTS WERE COLLECTED. SEE STACK TRACE FOR TERMINAL OUTPUT:" type=""><![CDATA[
EOF
)

cat $PWD/all_tests_formatted | while read line; do
    # Skipping extentions. Due to the failure:
    # ERROR: unknown command line flag 'gtest_list_tests'
    if [[ $line =~ "extensions/" || $line =~ "_yaml" ]]; then
      continue
    fi
    # addresses of the test directories are stored in the $line variable now
    echo "================================================================================"
    echo "RUNNING $line"
    echo "================================================================================"
    # list the tests here and then run them individually --gtest_list_tests
    ${PWD}/$line --gtest_list_tests |& tee test_cases
    test_first=$(head -n 1 test_cases)
    if [ "$test_first" == "Running main() from gtest_main.cc" ]; then
        tail -n +2 "test_cases" > "test_cases.tmp" && mv "test_cases.tmp" "test_cases"
    fi
    cat $PWD/test_cases | while read test_case_line; do
        last_char=${test_case_line: -1}
        if [ "$last_char" == "." ]; then
            test_suite=${test_case_line}
        else
            test_case="${test_suite}${test_case_line}"
            xml_file="gxf_engine_${test_case}.xml"
            counter_before=$(ls -1q ${BUILD_TESTLOG} | wc -l)
            timeout 120 ./$line --gtest_color=no --gtest_filter="$test_case" \
                --gtest_output="xml:${BUILD_TESTLOG}/" >gtest_std_output 2>gtest_error_msg
            gtest_exit_code=$?
            counter_after=$(ls -1q ${BUILD_TESTLOG} | wc -l)
            if [ "$counter_before" -eq "$counter_after" ]; then
                total_aborted=$((total_aborted + 1))
                msg1=$(<gtest_std_output)
                msg2=$(<gtest_error_msg)
                echo -e "${error_template_1}\n${msg1}\n${msg2}\n${error_template_2}" > ${xml_file}
                xml_string_manipulation ${xml_file} ${test_suite::-1} $test_case_line
            fi
        fi
    done
done

# JIRA tiket to refactor this: https://jirasw.nvidia.com/browse/ICICD-158
cat $PWD/python_tests_formatted | while read line; do
    # addresses of the test directories are stored in the $line variable now
    echo "================================================================================"
    echo "RUNNING PYTEST $line"
    echo "================================================================================"
    pytest_name="${line##*/}"
    xml_name="gxf_engine_${pytest_name}.xml"
    if [ -f "$PWD/gxf/python/gxf/core.py" ]; then
        cp "$PWD/gxf/python/gxf/core.py" "$PWD/gxf/core/__init__.py"
    fi
    PYTHONPATH=$PWD:$PWD/gxf/python python3 -m pytest \
        --junitxml=${BUILD_TESTLOG}/${xml_name} $line.py >std_output 2>error_msg
    exit_code=$?
    output=$(<std_output)
    echo $output
    if [ $exit_code -eq 2 ]; then
        msg=$(<error_msg)
        echo -e "${error_template_1}\n${msg}\n${error_template_2}" > ${xml_name}
        xml_string_manipulation ${xml_name} $pytest_name pytest
    fi
    if [ $exit_code -eq 5 ]; then
        msg=$(<std_output)
        echo -e "${error_template_3}\n{$msg}\n${error_template_2}" > ${xml_name}
        xml_string_manipulation ${xml_name} $pytest_name pytest
    fi
    rm -rf std_output
    rm -rf error_msg
done

throwErrors # failing a command if result is non-null

# change all the classnames here to Jetson.<classname>, to seperate from bazel tests
for file in $PWD/jetson_testlog/*
do
    sed -i 's/classname="/&Jetson\./g' $file
done
