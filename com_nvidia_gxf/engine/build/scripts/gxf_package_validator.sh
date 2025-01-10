#!/usr/bin/env bash
#####################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Source the helper functions
source "engine/build/scripts/utility_functions.sh"

change_to_workspace_dir

# Function to check if a file is in tarball_content.yaml
is_in_tarball() {
    local file="$1"
    local tarball_content_file="$2"
    log_message "Searching for: $file in $tarball_content_file${NC}" # Debug statement
    grep -q "$file" "$tarball_content_file"
}

# Function to check if a file exists recursively in the current directory
file_exists_recursively() {
    local file="$1"
    log_message "Searching for: $file in workspace${NC}" # Debug statement
    if find . -path "*/$file" -print -quit | grep -q .; then
        log_success "File found: $file${NC}"
        return 0 # File found
    else
        log_error "File not found: $file${NC}"
        return 1 # File not found
    fi
}

# Function to check if a file should be ignored
should_ignore_file() {
    local file="$1"
    [[ "$file" == *"_gen.yaml" ]]
}

# Function to check if a .so file should be ignored
should_ignore_so() {
    local file="$1"
    [[ "$file" == *.so ]] || [[ "$file" != *.lo ]] || [[ "$file" != gxe/gxe ]]
}

# Arrays to store files
declare -a missing_from_tarball
declare -a missing_from_tarball_build
declare -a missing_from_system

# Main script
main() {
    local parent_dir=$(pwd)
    local tarball_file="$parent_dir/release/tarball_content.yaml"

    if [ ! -f "$tarball_file" ]; then
        log_error "$tarball_file not found${NC}"
        exit 1
    fi

    # Process BUILD.release files
    while IFS= read -r -d '' build_file; do
        build_dir=$(dirname "$build_file")
        log_message "Checking $build_file${NC}"

        while IFS= read -r line; do
            [[ "$line" == *"{"* ]] && continue

            if [[ "$line" =~ \.(yaml|cpp|h|hpp|py) ]]; then
                file=$(echo "$line" | grep -oE '[^[:space:]"'\'']+\.(yaml|cpp|h|hpp|py)')

                # Skip files that should be ignored
                if should_ignore_file "$file"; then
                    log_warning "  [IGNORED] $file${NC}"
                    continue
                fi

                if [[ "$file" == /* ]]; then
                    relative_path="${file#/}"
                elif [[ "$file" == */* ]]; then
                    if [[ -f "$parent_dir/$file" ]]; then
                        relative_path="$file"
                    else
                        relative_path="${build_file%/*}/$file"
                        relative_path="${relative_path#$parent_dir/}"
                    fi
                else
                    relative_path="${build_file%/*}/$file"
                    relative_path="${relative_path#$parent_dir/}"
                fi

                relative_path="${relative_path#./}"

                if ! is_in_tarball "$relative_path" "$tarball_file"; then
                    log_warning "  [MISSING FROM TARBALL] $relative_path${NC}"
                    missing_from_tarball+=("$relative_path")
                    missing_from_tarball_build+=("${build_file#$parent_dir/}")
                else
                    log_success "  [OK] $relative_path${NC}"
                fi
            fi
        done < "$build_file"
    done < <(find . -name "BUILD.release" -print0)

    # Checking all the files from tarball_content, if they present in system
    log_message "Checking files from tarball_content.yaml${NC}"

    # Extract files from the relevant sections using yq
    files_to_check=$(yq -r '.files_to_copy_test_as_is[]' "$tarball_file")
    manifest_files=$(yq -r '.manifest[]' "$tarball_file")

    missing_from_system=()

    # Check files from files_to_copy_test_as_is section
    log_success "Checking files_to_copy_test_as_is section${NC}"
    for file in $files_to_check; do
        file=$(echo "$file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//') # Trim whitespace
        if [[ -n "$file" ]] && should_ignore_so "$file"; then
            if file_exists_recursively "$file"; then
                log_success "  [FOUND IN SYSTEM] $file${NC}"
            else
                log_error "  [MISSING FROM SYSTEM] $file${NC}"
                missing_from_system+=("$file")
            fi
        fi
    done

    # Check files from manifest section
    log_message "Checking manifest section${NC}"
    for file in $manifest_files; do
        file=$(echo "$file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//') # Trim whitespace

        if [[ -n "$file" ]] && ! should_ignore_so "$file"; then
            if file_exists_recursively "$file"; then
                log_success "  [FOUND IN SYSTEM] $file${NC}"
            else
                log_error "  [MISSING FROM SYSTEM] $file${NC}"
                missing_from_system+=("$file")
            fi
        fi
    done

    # Print summary and check for errors
    error_occurred=false

    # Print summary of files missing from tarball
    if [ ${#missing_from_tarball[@]} -gt 0 ]; then
        log_warning "Files Missing from Tarball:${NC}"
        log_warning "====================================${NC}"
        for i in "${!missing_from_tarball[@]}"; do
            log_warning "File: ${missing_from_tarball[$i]}${NC}"
            log_warning "BUILD.release: ${missing_from_tarball_build[$i]}${NC}"
            log_warning "----------------------------${NC}"
        done
        log_warning "Total files missing from tarball: ${#missing_from_tarball[@]}${NC}"
        error_occurred=false
    fi

    # Print summary of files missing from workspace
    if [ ${#missing_from_system[@]} -gt 0 ]; then
        log_error "Files in Tarball but Missing from System:${NC}"
        log_error "==================================================${NC}"
        for file in "${missing_from_system[@]}"; do
            log_error "File: $file${NC}"
            log_error "----------------------------${NC}"
        done
        log_error "Total files in tarball but missing from system: ${#missing_from_system[@]}${NC}"
        error_occurred=true
    fi

    if $error_occurred; then
        log_error "One or more checks failed. Please review the errors above.${NC}"
        exit 1
    else
        log_success "All checks passed successfully.${NC}"
        exit 0
    fi
}

main
