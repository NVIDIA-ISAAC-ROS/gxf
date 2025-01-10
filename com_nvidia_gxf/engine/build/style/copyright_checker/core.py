# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Core library for checking that copyright headers are correct.
"""

from typing import List, Tuple
import enum
import fnmatch
import pathlib
import re

# The number of lines that are extracted to check for the copyright header.
NUM_HEADER_LINES = 20

# Patterns for files that should be ignored from copyright checking.
IGNORED_FILE_PATTERNS = ('*.json', '*.png', '*.jpg', '*.pb', '*.pb.txt', '*.pod',
                         '*/locked_requirements.txt', '*/third_party/*', '*NOTICE*',
                         '*/package_list*.txt', '*/third_party_notice.txt', '*.min.js',
                         '*/pva_auth_allowlist', '*/LICENSE.txt')

# Constants to modify color of print statements.
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'


class CopyrightStatus(enum.Enum):
    OK = 1
    FIXED = 2
    MISSING_COPYRIGHT = 3
    WRONG_YEAR = 4


def is_ignored(path: pathlib.Path) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in IGNORED_FILE_PATTERNS)


def read_first_n_lines(path: pathlib.Path, num_lines: int) -> str:
    with open(path) as file:
        header = file.readlines()[:num_lines]
    return ''.join(header)


def replace_first_n_lines(path: pathlib.Path, num_lines: int, header: str):
    with open(path, 'r+') as file:
        lines = file.readlines()[num_lines:]
        file.seek(0)
        file.write(header)
        file.writelines(lines)


def has_copyright(header: str) -> bool:
    regex_query = r'.*copyright'
    match_object = re.match(regex_query, header, flags=re.IGNORECASE | re.DOTALL)
    return bool(match_object)


def has_correct_copyright_year(header: str, target_year: int) -> bool:
    regex_query = fr'.*copyright.*{target_year}'
    match_object = re.match(regex_query, header, flags=re.IGNORECASE | re.DOTALL)
    return bool(match_object)


def fix_copyright_year(header: str, target_year: int) -> str:
    # The regex should do the following:
    # "copyright 2018" -> "copyright 2018-2022"
    # "copyright 2018-2019" -> "copyright 2018-2022"
    # "copyright 2018-2022" -> "copyright 2018-2022"
    regex_match = r'(copyright (\(c\) )?[0-9]{4})(-[0-9]{4})?'
    regex_replace = fr'\1-{target_year}'
    fixed_header = re.sub(regex_match, regex_replace, header, flags=re.IGNORECASE)
    return fixed_header


def check_header(header: str, target_year: int, fix: bool) -> Tuple[CopyrightStatus, str]:
    if not has_copyright(header):
        return CopyrightStatus.MISSING_COPYRIGHT, header

    if has_correct_copyright_year(header, target_year):
        return CopyrightStatus.OK, header

    if not fix:
        # File has incorrect copyright but we are not allowed to fix it.
        return CopyrightStatus.WRONG_YEAR, header

    header = fix_copyright_year(header, target_year)
    if not has_correct_copyright_year(header, target_year):
        return CopyrightStatus.WRONG_YEAR, header

    return CopyrightStatus.FIXED, header


def check_file(path: pathlib.Path, target_year: int, fix: bool) -> bool:
    if not path.exists() or is_ignored(path):
        print(f'  - {path} [{GREEN}IGNORED{END}]')
        return True

    header = read_first_n_lines(path, NUM_HEADER_LINES)
    status, header = check_header(header, target_year, fix)
    if status == CopyrightStatus.FIXED:
        replace_first_n_lines(path, NUM_HEADER_LINES, header)

    if status in {CopyrightStatus.OK, CopyrightStatus.FIXED}:
        print(f'  - {path} [{GREEN}{status.name}{END}]')
        return True
    else:
        print(f'  - {path} [{RED}{status.name}{END}]')
        return False


def check_files(paths: List[pathlib.Path], target_year: int, fix: bool) -> bool:
    success = True
    print('Checking files:')
    for path in paths:
        if path.is_dir():
            raise Exception(f'Cannot check path "{path}" since it is a directory.')
        success &= check_file(path, target_year, fix)
    return success
