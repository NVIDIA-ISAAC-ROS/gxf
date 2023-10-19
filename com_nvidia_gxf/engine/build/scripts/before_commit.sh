#!/bin/bash
#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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

if ! type yapf &>/dev/null
then
  python3 -m pip install yapf --user
fi

GOBINARY=/usr/bin/go

if [ ! -f $GOBINARY ]
then
    echo "Go 1.14 not present, downloading..."
    curl -L -o /tmp/goinstall.tar.gz https://golang.org/dl/go1.14.6.linux-amd64.tar.gz
    sudo tar -xzf /tmp/goinstall.tar.gz -C /usr/local/
    sudo ln -s /usr/local/go/bin/go /usr/bin/go
fi


if [ -z "$GOPATH" ]
then
    echo "GOPATH is not set, using HOME/go by default..."
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

# Runs linter tests
bazel test ... --config lint
