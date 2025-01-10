# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Check that all passed files have a copyright header including the correct year.
"""
import argparse
import datetime
import pathlib
import sys

import core


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files',
                        nargs='+',
                        type=pathlib.Path,
                        help='The files that will be checked.')
    parser.add_argument('--year',
                        '-y',
                        default=datetime.date.today().year,
                        help='The target year that has to be in the copyright.')
    parser.add_argument('--fix-year',
                        '-f',
                        action='store_true',
                        help='Whether erroneous years should be auto-fixed.')
    args = parser.parse_args()

    success = core.check_files(args.files, args.year, args.fix_year)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
