#!/bin/bash
################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################
set -e

SCRIPT_DIR=$(dirname $(realpath $0))
GXE_PATH="${SCRIPT_DIR}/gxe"
REG_CLI_PATH="${SCRIPT_DIR}/registry"
USE_MANIFEST=""
RM_INSTALL_DIR=0
GRAPH_FILE=""
GRAPH_FILES_LIST=()
GRAPH_FILES_LIST_STR=""
RUN_PREFIX=""
TARGET=""
TARGET_RUN_PREFIX=""
RESOURCE_FILE=""
SUBGRAPHS=""
RUN_POSTFIX=""

function usage() {
    echo "Usage: $0 [options] <graph-file> [additional graph files]"
    echo ""
    echo "Options:"
    echo "  -d, --graph-target \"<graph-target-file>\"    [Required] Graph target config file"
    echo "  -s, --subgraphs <subgraph1>,<subgraph2>,... [Optional] Paths of subgraphs used by the application, comma-separated list"
    echo "      --resources <graph-resources-file>      [Optional] Graph resources file"
    echo "  -f, --fresh-manifest                        [Optional] Re-install graph and generate a new manifest file"
    echo "  -g, --with-gdb                              [Optional] Execute the graph under gdb"
    echo "  -m, --use-manifest <existing-manifest>      [Optional] Use an existing manifest file"
    echo "  -r, --remove-install-dir                    [Optional] Remove graph installation directory during exit"
    echo "  -t, --target <username@host>                [Optional] Target to execute the graph on. SSH will be used"
    echo "      --target-env-vars \"<env-vars>\"          [Optional] Separated list of environment variables to be set before running on target"
    echo "  -a  --app-root <app-root>                   [Optional] Root path for gxe to search subgraphs"
    echo ""
    echo "NOTE: To execute graphs on a remote target:"
    echo "* Graph Composer package needs to be already installed on the target"
    echo "* It is recommended that a password-less login method be used for SSH"
}

function log_error() {
    echo -e "\e[31m[ERROR] $@ \e[0m"
}

function log_info() {
    echo -e "\e[32m[INFO] $@ \e[0m"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ "$arg" == "-"* ]]; then
        case "$arg" in
        "--fresh-manifest")
            ;&
        "-f")
            FRESH_MANIFEST=1;;
        "--remove-install-dir")
            ;&
        "-r")
            RM_INSTALL_DIR=1;;
        "--use-manifest")
            ;&
        "-m")
            shift
            USE_MANIFEST="$1"
            if [ ! -f "$1" ]; then
                log_error "Invalid manifest file specified: $1"
                exit 1
            fi;;
        "--with-gdb")
            ;&
        "-g")
            RUN_PREFIX="gdb --args";;
        "--target")
            ;&
        "-t")
            shift
            if [ -z "$1" ]; then
                log_error "Target ssh details not provided"
                exit 1
            fi
            TARGET="$1"
            TARGET_RUN_PREFIX="ssh $1";;
        "--target-env-vars")
            shift
            if [ -z "$1" ]; then
                log_error "Target environment variables not provided"
                exit 1
            fi
            TARGET_ENV_VARS="$1";;
        "--graph-target")
            ;&
        "-d")
            shift
            GRAPH_TARGET="$1"
            if [ ! -f "$1" ]; then
                log_error "Invalid graph target file specified: $1"
                exit 1
            fi;;
        "--resources")
            shift
            RESOURCE_FILE="$1"
            if [ ! -f "$1" ]; then
                log_error "Invalid graph resource file specified: $1"
                exit 1
            fi;;
        "--app-root")
            ;&
        "-a")
            shift
            if [ ! -d "$1" ]; then
                log_error "Graph app root is not valid: $1"
                exit 1
            fi
            RUN_POSTFIX="-app_root $1";;
        "--subgraphs")
            ;&
        "-s")
            shift
            SUBGRAPHS="$1"
            if [ -z "$1" ]; then
                log_error "Subgraphs not specified"
                exit 1
            fi;;
        *) log_error "Unknown argument ${arg}"; usage; exit 1
        esac
    else
        if [ ! -f "${arg}" ]; then
            log_error "Invalid graph file specified: ${arg}"
            exit 1
        fi
        if [[ -z "${GRAPH_FILE}" ]]; then
            GRAPH_FILE="$(realpath $arg)"
	    GRAPH_FILES_LIST_STR="$(basename $GRAPH_FILE)"
	else
            GRAPH_FILES_LIST+=($(realpath "$1"))
	    GRAPH_FILES_LIST_STR="$GRAPH_FILES_LIST_STR,$(basename $1)"
        fi
    fi
    shift
done

if [[ -z "${GRAPH_FILE}" ]]; then
    log_error "No graph file specified"
    usage
    exit 1
fi
if [[ -z "${GRAPH_TARGET}" ]]; then
    log_error "No graph target file specified"
    usage
    exit 1
fi

if [[ ! -z "${TARGET}" ]]; then
    log_info "Trying to access target '$TARGET' over ssh"

    set +e
    ssh -q -o StrictHostKeyChecking=no -o PubkeyAuthentication=yes \
        -o PasswordAuthentication=no "${TARGET}" "ls >/dev/null </dev/null"
    if [ $? -ne 0 ]; then
        log_info "***SSH password-less login not set up. Script will ask for password multiple times. Copy key to target using 'ssh-copy-id ${TARGET}'***"
    fi

    set -e
    if ! ${TARGET_RUN_PREFIX} "whoami" >/dev/null 2>&1; then
        log_error "Could not access target"
        exit 1
    fi

    if ! which sshfs >/dev/null; then
        log_error "Running remotely on target requires the 'sshfs' package be install on this machine."
        exit 1
    fi

    if ! ${TARGET_RUN_PREFIX} ls ${GXE_PATH} >/dev/null 2>&1; then
        log_error "Graph composer not found on target"
        exit 1
    fi

    log_info "Target access successful"
fi

ARCH="$($TARGET_RUN_PREFIX arch)"

echo "Graphs: ${GRAPH_FILES_LIST_STR}"
echo "Target: ${GRAPH_TARGET}"
echo "==================================================================="
echo "Running $(basename ${GRAPH_FILE})"
echo "==================================================================="

function print_end() {
    echo "*******************************************************************"
    echo "End $(basename ${GRAPH_FILE})"
    echo "*******************************************************************"
    if [[ ${RM_INSTALL_DIR} -eq 1 ]]; then
        ${TARGET_RUN_PREFIX} rm -rf "${MANIFEST_DIR}"
        log_info "Graph installation directory ${MANIFEST_DIR} and manifest ${MANIFEST_FILE} removed"
    else
        log_info "Graph installation directory ${MANIFEST_DIR} and manifest ${MANIFEST_FILE} retained"
    fi

    if [[ ! -z "${TARGET}" ]]; then
        fusermount -u "${MANIFEST_DIR_TARGET}" 2>/dev/null || true
    fi
}
trap print_end EXIT

function mount_target_dir() {
    mkdir -p "${MANIFEST_DIR_TARGET}"
    ${TARGET_RUN_PREFIX} mkdir -p "${MANIFEST_DIR}"
    sshfs "${TARGET}:${MANIFEST_DIR}" "${MANIFEST_DIR_TARGET}"
}

function copy_file_to_target() {
    FSRC="$1"
    FTARGET="$2"
    FTARGET_DIR="$(dirname ${FTARGET})"
    if [[ "${FSRC}" == "None" ]]; then
        FSRC=${FTARGET}
    fi
    if [[ ! -d "${FSRC}" && ! -f "${FSRC}" ]]; then
        log_error "Could not find '${FSRC}' on host"
        return 1
    fi
    if [[ "${FTARGET}" == "/"* ]]; then
        if ! ${TARGET_RUN_PREFIX} mkdir -p "${FTARGET_DIR}"; then
            log_error "Failed to create directory '${FTARGET_DIR}' on target"
            return 1
        fi
        if ! rsync -r "${FSRC}" "${TARGET}:${FTARGET}"; then
            log_error "Failed to copy from '${FSRC}' to '${FTARGET}' on target"
            return 1
        fi
    else
        if ! mkdir -p "${MANIFEST_DIR_TARGET}/${FTARGET_DIR}"; then
            log_error "Failed to create directory '${FTARGET_DIR}' in graph install directory"
            return 1
        fi
        if ! rsync -r "${FSRC}" "${MANIFEST_DIR_TARGET}/${FTARGET}"; then
            log_error "Failed to copy from '${FSRC}' to '${FTARGET}' in graph install directory"
        fi
    fi
}

if [[ -z "$USE_MANIFEST" ]]; then
    MANIFEST_DIR="/tmp/ds.$(basename ${GRAPH_FILE} | sed -e 's/\.[^.]*$//')"
    if [[ ${FRESH_MANIFEST} -eq 1 ]]; then
        if ${TARGET_RUN_PREFIX} ls "${MANIFEST_DIR}" >/dev/null 2>&1; then
            log_info "Removing existing installation : ${MANIFEST_DIR}"
	    ${TARGET_RUN_PREFIX} rm -rf "${MANIFEST_DIR}"
        fi
        log_info "Generating fresh manifest file in installation directory ${MANIFEST_DIR}"
        ${TARGET_RUN_PREFIX} mkdir "${MANIFEST_DIR}"
    fi
    MANIFEST_FILE="${MANIFEST_DIR}/manifest.yaml"
    log_info "Writing manifest to ${MANIFEST_FILE}"
else
    MANIFEST_FILE="${USE_MANIFEST}"
    log_info "Using existing manifest ${MANIFEST_FILE}"
fi

MANIFEST_DIR_TARGET="${MANIFEST_DIR}"
MANIFEST_FILE_TARGET="${MANIFEST_FILE}"
if [[ ! -z "${TARGET}" ]]; then
    MANIFEST_DIR_TARGET="${MANIFEST_DIR}-${TARGET}"
    MANIFEST_FILE_TARGET="${MANIFEST_DIR_TARGET}/manifest.yaml"
    mount_target_dir
    cp ${GRAPH_FILE} ${MANIFEST_DIR_TARGET}
    for gph in ${GRAPH_FILES_LIST[@]}; do
        cp $gph ${MANIFEST_DIR_TARGET}
    done
    if [[ ! -z "${RESOURCE_FILE}" ]]; then
        if ! which shyaml >/dev/null; then
            log_error "'shyaml' python package not found"
            exit 1
        fi
        if ! which rsync >/dev/null; then
            log_error "'rsync' apt package not installed"
            exit 1
        fi
        log_info "Copying file resources..."
        NUM_FILES="$(cat ${RESOURCE_FILE} | shyaml get-length files 2>/dev/null || echo 0)"
        I=0
        while [[ $I -lt ${NUM_FILES} ]]; do
            FSRC="$(cat ${RESOURCE_FILE} | shyaml get-value files.$I.src)"
            FTARGET="$(cat ${RESOURCE_FILE} | shyaml get-value files.$I.target)"
            copy_file_to_target "${FSRC}" "${FTARGET}"
            I=$((I + 1))
        done
    fi
fi

SUBGRAPHS="$(echo $SUBGRAPHS | sed 's/,/ /g')"

if [[ ! -f "$MANIFEST_FILE_TARGET" ]]; then
    ${REG_CLI_PATH} graph install -g "${GRAPH_FILE}" ${SUBGRAPHS} \
        -m "${MANIFEST_FILE_TARGET}" --target-file-path "${GRAPH_TARGET}" \
        --output-directory "${MANIFEST_DIR_TARGET}" || exit 1
    if [[ ! -z "${TARGET}" ]]; then
        sed -i "s/-${TARGET}//" "${MANIFEST_FILE_TARGET}"
    fi
fi

if [[ ! -z "$TARGET" ]]; then
    GRAPH_FILES="${MANIFEST_DIR}/$(basename ${GRAPH_FILE})"
    for gph in ${GRAPH_FILES_LIST[@]}; do
        GRAPH_FILES="${GRAPH_FILES},${MANIFEST_DIR}/$(basename $gph)"
    done
    ${TARGET_RUN_PREFIX} -t "cd ${MANIFEST_DIR}; ${TARGET_ENV_VARS} ${RUN_PREFIX} \
        ${GXE_PATH} -app ${GRAPH_FILES} -manifest ${MANIFEST_FILE} ${RUN_POSTFIX}" || exit 1
else
    GRAPH_FILES="${GRAPH_FILE}"
    for gph in ${GRAPH_FILES_LIST[@]}; do
        GRAPH_FILES="${GRAPH_FILES},$gph"
    done
    ${RUN_PREFIX} ${GXE_PATH} -app "${GRAPH_FILES}" \
        -manifest "${MANIFEST_FILE}" ${RUN_POSTFIX} || exit 1
fi
