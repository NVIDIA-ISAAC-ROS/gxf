#!/bin/bash
#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
set +e

WORK_DIR=/tmp/minidump
SYMBOL_DIR=$WORK_DIR/syms

if [ $# -lt 1 ]; then
    printf "Usage: process_syms <executable_file>\n"
    exit
fi
mkdir -p $SYMBOL_DIR

EXECUTABLE_FILE="$1"

# Extracts symbols and wait until it finishes
SYM_FILE=$(mktemp)
$WORK_DIR/dump_syms $EXECUTABLE_FILE > $SYM_FILE &2>/dev/null && wait

# Moves symbol file to folder structure based on its hash and binary name
HASH=$(head -n1 $SYM_FILE | awk '{print $4}')
BINARY_NAME=$(head -n1 $SYM_FILE | awk '{print $5}')
SYM_DEST_DIR=$SYMBOL_DIR/$BINARY_NAME/$HASH
SYM_FULL_PATH=$SYM_DEST_DIR/$BINARY_NAME.sym
mkdir -p $SYM_DEST_DIR
mv $SYM_FILE $SYM_FULL_PATH