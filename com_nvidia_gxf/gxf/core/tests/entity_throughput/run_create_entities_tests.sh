#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

LOGFILE="test_create_entities.log"

# The array of thread numbers to be used
THREAD_NUMS=(1 2 4 8)

# Path to the .cpp file to modify
SOURCE_FILE="./test_create_entities.cpp"

# Backup original file
cp $SOURCE_FILE "${SOURCE_FILE}.backup"

# Function to replace the NumThread value and run the tests
run_tests_for_thread_num() {
    local num_thread=$1
    # Use sed to substitute "const int NumThread =;" with the desired thread number
    sed -i "s/const int kNumThread = 1;/const int kNumThread = $num_thread;/" $SOURCE_FILE

    # Optional: display current thread number in logfile
    echo -e "\nRunning tests with NumThread = $num_thread\n" | tee -a $LOGFILE

    # Run the bazel command 5 times
    for i in {1..5}
    do
        echo "Running test $i for NumThread = $num_thread ..." | tee -a $LOGFILE
        bazel run //gxf/core/tests/entity_throughput:test_entity_create 2>&1 | tee -a $LOGFILE
    done

    # Restore the original .cpp file
    cp "${SOURCE_FILE}.backup" $SOURCE_FILE
}

# Loop through each thread number, update the .cpp file, and run the tests
for num in "${THREAD_NUMS[@]}"
do
    run_tests_for_thread_num $num
done

# Restore the original .cpp file
mv "${SOURCE_FILE}.backup" $SOURCE_FILE