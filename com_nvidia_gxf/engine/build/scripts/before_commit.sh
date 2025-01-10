#!/bin/bash
#####################################################################################
# Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# This script would format Bazel BUILD files and C++ source files that are touched in latest commit.

set -e

#if ! type clang-format 2>/dev/null
#then
#  sudo apt install clang-format -y
#fi

# Source the helper functions
source "engine/build/scripts/utility_functions.sh"

# Function to check if a file is staged for commit
is_file_pattern_staged() {
    git diff --cached --name-only | grep -q "$1"
}

if ! type yapf &>/dev/null
then
  python3 -m pip install yapf --user
fi

GOBINARY=/usr/bin/go

if [ ! -f $GOBINARY ]
then
    log_message "Go 1.14 not present, downloading..."
    curl -L -o /tmp/goinstall.tar.gz https://golang.org/dl/go1.14.6.linux-amd64.tar.gz
    sudo tar -xzf /tmp/goinstall.tar.gz -C /usr/local/
    sudo ln -s /usr/local/go/bin/go /usr/bin/go
fi


if [ -z "$GOPATH" ]
then
    log_message "GOPATH is not set, using HOME/go by default..."
    export GOPATH=$HOME/go/
fi

BUILDIFIER_BINARY=$GOPATH/bin/buildifier

if ! type $BUILDIFIER_BINARY 2>/dev/null
then
  $GOBINARY get github.com/bazelbuild/buildtools/buildifier
fi

DIR=$( git rev-parse --show-toplevel )

cd $DIR

TOUCHED="$(git diff HEAD^ --name-only)"

for f in $TOUCHED
do
  if [ ! -f $f ]; then
      continue
  fi

  ext=${f##*.}
  fn=$(basename $f)
  base=${fn%.*}
  if [ "$base" = "BUILD" ] || [ "$base" = "WORKSPACE" ] || [ "$ext" = "bzl" ] || [ "$ext" = "BUILD" ]
  then
      $BUILDIFIER_BINARY $f
  fi
  if [ "$ext" = "py" ]
  then
      yapf -i $f
  fi
#  if [ "$ext" = "h" ] || [ "$ext" = "hpp" ] || [ "$ext" = "c" ] || [ "$ext" = "cc" ] || [ "$ext" = "cpp" ] || [ $ext = "cxx" ];
#  then clang-format -i $f
#  fi
done

STAGED="$(git diff --staged --name-only --diff-filter=d)"
UNSTAGED="$(git diff --name-only)"

STAGED_AND_UNSTAGED_FILES="$(echo "$UNSTAGED $STAGED" | tr ' ' '\n' | grep -v '^$' | sort | uniq -d)"
if [ -n "${STAGED_AND_UNSTAGED_FILES}" ]; then
  log_error "Error: Cannot format commit because it contains the following partially staged files:"
  sed -E 's/^/  - /' <<< "$STAGED_AND_UNSTAGED_FILES"
  log_error "Your options are:"
  log_error '  - Stash the unstaged files: `git stash -k`' and commit again
  log_error '  - Override the pre-commit hook: `git commit --no-verify` (only if you are sure that your code is formatted properly)'
  exit 1
fi

if [ -z "$STAGED" ]; then
  log_error "No staged files found."
  exit 1
fi

check_and_format_copyright() (
  log_message "Checking copyright..."
  local staged_files="${1?}"

  # Make the file paths absolute.
  staged_files=$(awk "{print \"$(pwd)/\" \$0}" <<< $staged_files)
  quiet_flags="--test_output=all"
  bazel run $quiet_flags //engine/build/style/copyright_checker:check_files -- --fix-year $staged_files
)

check_and_format_copyright "${STAGED}" || (echo "Error: Copyright check failed."; exit 1)

# Runs linter tests
bazel test ... --config lint

log_message "Running gxf_package_validator.sh..."
if ! ./engine/build/scripts/gxf_package_validator.sh; then
    log_error "gxf_package_validator.sh script failed. Please check the output and fix any issues."
    exit 1
fi
log_success "gxf_package_validator.sh completed successfully."

# If all checks pass, allow the commit
log_success "All checks passed. Proceeding with commit."
exit 0
