# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Check that all files in a git repo, that were modified in a specific year have a copyright header
including the correct year.
"""
from typing import List
import argparse
import datetime
import os
import pathlib
import sys

import git

import core


def get_default_repo_path() -> str:
    # If run from bazel we want to use the workspace directory, else the current working directory.
    repo_path = os.environ.get('BUILD_WORKING_DIRECTORY', os.getcwd())
    return repo_path


def get_commits_in_year(repo: git.Repo, year: int) -> List[str]:
    commits = repo.git.log('--reverse', '--date-order', f'--since={year}-01-01',
                           f'--until={year}-12-31', '--format=%H').split('\n')
    return commits


def get_changed_files_between_commits(repo: git.Repo, commit_a: str, commit_b: str) -> List[str]:
    files = repo.git.diff('--name-only', f'{commit_a}...{commit_b}', '--diff-filter=d').split('\n')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repo',
                        '-r',
                        default=get_default_repo_path(),
                        help='Path to the repo that should be checked.')
    parser.add_argument('--year',
                        '-y',
                        default=datetime.date.today().year,
                        help='All files modified in this year will be checked.')
    parser.add_argument('--fix-year',
                        '-f',
                        action='store_true',
                        help='Whether erroneous years should be auto-fixed.')
    args = parser.parse_args()

    repo = git.Repo(args.repo, search_parent_directories=True)
    commits = get_commits_in_year(repo, args.year)
    files = get_changed_files_between_commits(repo, commits[0], commits[-1])
    files = [pathlib.Path(x) for x in files]

    # Change working dir to repo path because file paths are relative to repo path.
    os.chdir(repo.working_tree_dir)
    print(f'Checking all files in repo {repo.working_tree_dir} that were modified in {args.year}.')
    success = core.check_files(files, args.year, fix=args.fix_year)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
