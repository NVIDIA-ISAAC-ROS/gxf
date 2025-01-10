# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import re

# Define the regex pattern to extract the relevant information
pattern = re.compile(r"TIME TAKEN (\d+) ms, threadNum: (\d+)")

# Dictionaries to store accumulated times and count for each threadNum
times = {}
counts = {}

# Read the log file and process each line
with open('./test_create_entities.log', 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            time_taken, thread_num = int(match.group(1)), int(match.group(2))

            # Update accumulated time and count for the current threadNum
            times[thread_num] = times.get(thread_num, 0) + time_taken
            counts[thread_num] = counts.get(thread_num, 0) + 1

# Calculate and print averages for each threadNum
for thread_num, total_time in times.items():
    avg_time = total_time / counts[thread_num]
    entity_per_sec = thread_num * 100000000 / avg_time
    print(f"ThreadNum {thread_num}: TIME TAKEN = {avg_time:.2f} ms, entities/s = {entity_per_sec:.2f}")

