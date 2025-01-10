#!/usr/bin/env bash
#####################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')][SUCCESS] $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')][ERROR] $1${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')][WARNING] $1${NC}"
}

# Helper function to change to workspace directory
change_to_workspace_dir() {
    if [ -n "$BUILD_WORKSPACE_DIRECTORY" ]; then
        cd "$BUILD_WORKSPACE_DIRECTORY" || {
            log_error "Failed to change directory to $BUILD_WORKSPACE_DIRECTORY"
            return 1
        }
        log_success "Changed working directory to: $BUILD_WORKSPACE_DIRECTORY"
    else
        log_warning "BUILD_WORKSPACE_DIRECTORY is not defined. Running in current directory."
    fi
}

# Export functions so they can be used in other scripts
export -f log_message
export -f log_success
export -f log_error
export -f log_warning
export -f change_to_workspace_dir

# Check if we are being sourced or executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    log_warning "This script is meant to be sourced, not executed directly."
fi
