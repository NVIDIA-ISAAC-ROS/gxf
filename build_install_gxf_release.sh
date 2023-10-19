#!/bin/bash -e
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Rebuild GXF source and installs release binaries into isaac_ros_gxf package.

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Constants
PATCHES_PATH="${ROOT}/patch"
GXF_REPO_PATH="${ROOT}/com_nvidia_gxf"

RELEASE_TARBALL_DIR="${ROOT}"
RELEASE_TARBALL_FILE="gxf_isaac_release.tar.gz"

SUPPORTED_PLATFORMS=("x86_64_cuda_11_8" "jetpack51")

# Arguments
ISAAC_ROS_GXF_ROOT="${ROOT}/../../../ros_ws/src/isaac_ros_nitros/isaac_ros_gxf"
TARGET_GXF_DIR="${ISAAC_ROS_GXF_ROOT}/gxf/core"

# Print utils
# Color chart: https://linux.101hacks.com/ps1-examples/prompt-color-using-tput/
function print_color {
    tput setaf $1
    echo "$2"
    tput sgr0
}

function print_error {
    print_color 1 "$1"
}

function print_warning {
    print_color 3 "$1"
}

function print_info {
    print_color 2 "$1"
}

function error_and_exit {
    print_error "Error: $1"
    exit 1
}

# On exit trap
ON_EXIT=()
function cleanup {
    for command in "${ON_EXIT[@]}"
    do
        $command &>/dev/null
    done
}
trap cleanup EXIT

pushd . &>/dev/null
ON_EXIT+=("popd")


# Get command line arguments
OPTIONS=i:dh
LONGOPTS=install-path,dry-run,help
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
  case "$1" in
    -i|--install-path)
      ISAAC_ROS_GXF_ROOT="$2"
      shift 2
      ;;
    -d|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      print_error "$0: Unknown argument $1"
      exit 1
      ;;
  esac
done

# Help
function usage {
  print_info "Usage: build_install_gxf_release.sh -i {path to isaac_ros_gxf package}"
  print_info "Copyright (c) 2023, NVIDIA CORPORATION."
}

if [[ ! -d "$GXF_REPO_PATH" ]]; then
  print_error "GXF repository not found ($GXF_REPO_PATH)"
  usage
  exit 1
fi

if [[ ! -d "$ISAAC_ROS_GXF_ROOT" ]]; then
  print_error "isaac_ros_gxf package not found, please set with -i as path to isaac_ros_gxf package ($ISAAC_ROS_GXF_ROOT)"
  usage
  exit 1
fi

# Filenames of all the patches
PATCH_FILEPATHS=()
if [[ -d "$PATCHES_PATH" ]]; then
  PATCH_FILEPATHS=( `find ${PATCHES_PATH} -name "*.patch"` )
  for PATCH in "${PATCH_FILEPATHS[@]}"
  do
    print_info "Found GXF patch: ${PATCH}"
  done
fi

print_info "Applying ${#PATCH_FILEPATHS[@]} patches to GXF core"
cd ${GXF_REPO_PATH}
for PATCH in "${PATCH_FILEPATHS[@]}"
do
  print_info "Applying GXF patch: ${PATCH}"
  patch -p0 -f -F 5 -i ${PATCH}
done
print_info "Finished applying ${#PATCH_FILEPATHS[@]} patches to GXF core"

# Edit tarball contents
print_info "Generating tarball contents YAML configuration."
TARBALL_YAML_PATH="${ROOT}/build_gxf_release_content.yaml"
print_info "Tarball YAML path: ${TARBALL_YAML_PATH}"

TMPD=/tmp/gxf-release
print_info "Building GXF release into ${TMPD}"
rm -Rf ${TMPD}
mkdir -p ${TMPD}
ON_EXIT+=("rm -rf $TMPD")
python3 release/make_tarball.py ${TARBALL_YAML_PATH} ${RELEASE_TARBALL_FILE} ${TMPD}/

print_info "Moving result from ${GXF_REPO_PATH} into destination directory at ${RELEASE_TARBALL_DIR}"
mv ${GXF_REPO_PATH}/${RELEASE_TARBALL_FILE} ${RELEASE_TARBALL_DIR}/${RELEASE_TARBALL_FILE}
ON_EXIT+=("rm -rf ${RELEASE_TARBALL_DIR}/${RELEASE_TARBALL_FILE}")

print_info "Completed building."


print_info "Installing GXF release."
TMPD=$(mktemp -d)   # working directory
ON_EXIT+=("rm -rf $TMPD")
print_info "Expanding GXF release tarball to temp dir ${TMPD}"
tar -xzvf ${RELEASE_TARBALL_DIR}/${RELEASE_TARBALL_FILE} --directory ${TMPD}/

print_info "Removing existing GXF framework files in ${TARGET_GXF_DIR}"
rm -rf "${TARGET_GXF_DIR}"
mkdir -p "${TARGET_GXF_DIR}"

print_info "Installing GXF framework files in ${TARGET_GXF_DIR}"
rsync -qrvm \
--exclude "gxf_*/" \
--exclude "sample/" \
--exclude "gxf/sample/" \
--exclude "gxf/python_codelet/" \
--exclude "test/" \
--exclude "*.sh" \
--exclude "*.py" \
--exclude "*.cpp" \
--exclude "*.bzl" \
--exclude "*.yaml" \
--exclude "BUILD" \
--exclude "*.BUILD" \
--include "*/" \
--include "LICENSE*" \
--include "*.h" \
--include "*.hpp" \
--exclude "*" \
"${TMPD}/tmp/gxf-release/" "${TARGET_GXF_DIR}/include" || error_and_exit "Could not deploy GXF includes to ${TARGET_GXF_DIR}/include"

for PLATFORM in "${SUPPORTED_PLATFORMS[@]}"
do
mkdir -p "${TARGET_GXF_DIR}/lib/gxf_${PLATFORM}"
rsync -qrvm \
  --include "*/" \
  --include "*.so" \
  --exclude "*" \
  "${TMPD}/tmp/gxf-release/gxf_${PLATFORM}/" "${TARGET_GXF_DIR}/lib/gxf_${PLATFORM}" || error_and_exit "Could not deploy GXF ${PLATFORM} binaries to ${TARGET_GXF_DIR}/lib"
done

# Patchelf SONAMEs into shared libraries for name resolution
print_info "Patching SONAME into GXF shared libraires"
SOLIB_FILEPATHS=( `find ${TARGET_GXF_DIR} -name "libgxf_*.so"` )
for SOLIB_FILEPATH in "${SOLIB_FILEPATHS[@]}"
do
  SONAME=${SOLIB_FILEPATH##*/}
  chmod +w ${SOLIB_FILEPATH}
  patchelf --set-soname ${SONAME} ${SOLIB_FILEPATH}
done
print_info "Done patching SONAME into GXF shared libraires"


print_info "Completed. Installed rebuilt GXF framework to ${TARGET_GXF_DIR}"


