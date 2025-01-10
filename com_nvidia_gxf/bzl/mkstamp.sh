#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

echo BUILD_DATE $(TZ=Etc/UTC date -Iseconds)

IS_NIGHTLY=$(echo $NIGHTLY_TRIGGERED_JOB)
IS_NIGHTLY_APPS=$(echo $NIGHTLY_CONTAINER_PHASE)
IS_DEVELOPMENT=$(echo $DEVELOPMENT_TRIGGERED_JOB)
IS_CHECKOUT=$(git rev-parse --is-inside-work-tree 2> /dev/null || echo "false")

if [ "${IS_NIGHTLY}" == "true" ]
then
    HASH=$(git rev-parse HEAD)
    SHORT_HASH=$(git rev-parse --short HEAD)

    echo BUILD_SCM_HASH "${HASH}"
    echo BUILD_SCM_HASH_SHORT "${SHORT_HASH}"

    if [ "${IS_NIGHTLY_APPS}" == "true" ]
    then
        echo STABLE_SCM_HASH_SHORT "${SHORT_HASH}"
    else
        echo STABLE_SCM_HASH_SHORT CI_STABLE
    fi

    echo BUILD_SCM_DIRTY 0
    echo BUILD_SCM_DIVERGED 0
    echo BUILD_BRANCH_TAG "${BRANCH_NAME}"
elif [ "${IS_DEVELOPMENT}" == "true" ]
then
    HASH=$(git rev-parse HEAD)
    SHORT_HASH=$(git rev-parse --short HEAD)

    echo BUILD_SCM_HASH CI_STABLE
    echo BUILD_SCM_HASH_SHORT CI_STABLE
    echo STABLE_SCM_HASH_SHORT CI_STABLE
    echo BUILD_SCM_DIRTY 0
    echo BUILD_SCM_DIVERGED 0
    echo BUILD_BRANCH_TAG "${BRANCH_NAME}"
elif [ "${IS_CHECKOUT}" == "true" ]
then
    HASH=$(git rev-parse HEAD)
    SHORT_HASH=$(git rev-parse --short HEAD)

    echo BUILD_SCM_HASH "${HASH}"
    echo BUILD_SCM_HASH_SHORT "${SHORT_HASH}"
    echo STABLE_SCM_HASH_SHORT "${SHORT_HASH}"

    # outputs whether work tree is commited or not
    DIRTY=$([[ -z "$(git status --porcelain)" ]] && echo false || echo true)

    # get upstream branch
    UPSTREAM_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || git rev-parse --short HEAD)

    # whether local branch corresponds to upstream branch or contains additional commits
    DIVERGED=$([[ -z "${UPSTREAM_BRANCH}" || -n "$(git diff ${UPSTREAM_BRANCH})" ]] && echo true || echo false)

    # whether we have local modifications
    echo BUILD_SCM_DIRTY $(${DIRTY} && echo "1" || echo "0")

    # whether local is any different from upstream
    echo BUILD_SCM_DIVERGED $([[ ${DIRTY} || ${DIVERGED} ]] && echo "1" || echo "0" )

    TAG_BRANCH=$([[ ${DIRTY} || ${DIVERGED} || -z "${UPSTREAM_BRANCH}" ]] && echo "$SHORT_HASH" || echo "${UPSTREAM_BRANCH}")
    TAG_DIRTY=$(${DIRTY} && echo "-dirty" || echo "" )

    # full tag to apply to images etc.
    echo BUILD_BRANCH_TAG $(printf "%s%s" "${TAG_BRANCH}" "${TAG_DIRTY}")
else
    UUID=$(uuidgen)

    echo BUILD_SCM_HASH "${UUID}"
    echo BUILD_SCM_HASH_SHORT "${UUID}"
    echo STABLE_SCM_HASH_SHORT "${UUID}"
    echo BUILD_SCM_DIRTY 0
    echo BUILD_SCM_DIVERGED 0
    echo BUILD_BRANCH_TAG norepo-${UUID}
fi
