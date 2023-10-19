################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

#!/bin/bash

script="release_build_setup.sh"

#Number of mandatory arguments
margs=1

# Common functions - BEGIN
function example {
    echo -e "example: $script -p /tmp/packaging/gxf_x86_64/core/libgxf_core.so"
}

function usage {
    echo -e "usage: $script -p MANDATOR_PATH_TO_RELEASE_BUILD_LIB_GXF_CORE.SO\n"
}

function help {
  usage
    echo -e "MANDATORY:"
    echo -e "  -p,  --mandatory path   Path to libgxf_core.so in release installation folder. Example: <path_to_release_installation_folder>/gxf_x86_64/core/libgxf_core.so"
    echo -e "  -h,  --help             Prints this help\n"
  example
}

# Ensures that the number of passed args are at least equals
# to the declared number of mandatory args.
# It also handles the special case of the -h or --help arg.
function margs_precheck {
    if [ $2 ] && [ $1 -lt $margs ]; then
        if [ $2 == "--help" ] || [ $2 == "-h" ]; then
            help
            exit
        else
        	usage
            example
        	exit 1 # error
        fi
    fi
}

# Ensures the file path validity
function margs_check {

    if [ $# -lt $margs ]; then
        usage
      	    example
        exit 1 # error

    elif [ -f $marg0 ]
    then
        echo "File $marg0 exists."
    else
        echo "Error: File $marg0 does not exist."
        usage
        exit 1 # error
    fi
}
# Common functions - END

# Main
margs_precheck $# $1

# Args while-loop
while [ "$1" != "" ];
do
   case $1 in
   -p  | --mandatory )  shift
                          marg0=$1
                		  ;;

   -h   | --help )        help
                          exit
                          ;;
   *)
                          echo "$script: illegal option $1"
                          usage
                          example
                          exit 1 # error
                          ;;
    esac
    shift
done

# Check for the path validity
margs_check $marg0

# update the link for libgxf_core.so
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/libgxf_core.so gxf_core $marg0 55
