# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Check that all files modified in a commit have a copyright header including the correct year.
"""
import argparse
import os
import pathlib
import sys
import time

import git

import core


def get_default_repo_path() -> str:
    # If run from bazel we want to use the workspace directory, else the current working directory.
    repo_path = os.environ.get('BUILD_WORKING_DIRECTORY', os.getcwd())
    return repo_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repo',
                        '-r',
                        default=get_default_repo_path(),
                        help='Path to the repo that should be checked.')
    parser.add_argument('--commit',
                        '-c',
                        default='HEAD',
                        help='Hash of the commit that should be checked.')
    parser.add_argument('--fix-year',
                        '-f',
                        action='store_true',
                        help='Whether erroneous years should be auto-fixed.')
    args = parser.parse_args()

    repo = git.Repo(args.repo, search_parent_directories=True)
    commit = repo.commit(args.commit)
    commit_year = time.gmtime(commit.committed_date).tm_year
    files = [pathlib.Path(x) for x in commit.stats.files]

    # Change working dir to repo path because file paths are relative to repo path.
    os.chdir(repo.working_tree_dir)
    print(f'Checking commit {commit.hexsha} in repo {repo.working_tree_dir}')
    success = core.check_files(files, commit_year, fix=args.fix_year)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
