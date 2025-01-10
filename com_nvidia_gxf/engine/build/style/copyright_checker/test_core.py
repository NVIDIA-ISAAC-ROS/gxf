# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Tests for the core library to check the copyright headers.
"""

import unittest

import core

# Year that the fixed copyright headers should include.
TARGET_YEAR = 2022


def check_header_helper(header: str, target_year: int, fix: bool) -> bool:
    status, _ = core.check_header(header, target_year, fix)
    return status == core.CopyrightStatus.OK


class CheckCopyrightTest(unittest.TestCase):

    def setUp(self):
        self.wrong_1 = 'Copyright 2019, NVIDIA CORPORATION. All rights reserved.'
        self.wrong_2 = 'Copyright 2019-2021, NVIDIA CORPORATION. All rights reserved.'
        self.wrong_3 = 'Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.'
        self.wrong_4 = 'Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.'
        self.wrong_5 = \
        '''
        """
        Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

        NVIDIA CORPORATION and its licensors retain all intellectual property
        and proprietary rights in and to this software, related documentation
        and any modifications thereto. Any use, reproduction, disclosure or
        distribution of this software and related documentation without an express
        license agreement from NVIDIA CORPORATION is strictly prohibited.
        """

        load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_cc_library")
        '''
        self.wrong_6 = 'Hello World 2022.'

        self.correct_1 = 'Copyright 2022, NVIDIA CORPORATION. All rights reserved.'
        self.correct_2 = 'Copyright 2019-2022, NVIDIA CORPORATION. All rights reserved.'
        self.correct_3 = 'Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.'
        self.correct_4 = 'Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.'
        self.correct_5 = \
        '''
        """
        Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

        NVIDIA CORPORATION and its licensors retain all intellectual property
        and proprietary rights in and to this software, related documentation
        and any modifications thereto. Any use, reproduction, disclosure or
        distribution of this software and related documentation without an express
        license agreement from NVIDIA CORPORATION is strictly prohibited.
        """

        load("@com_nvidia_gxf//gxf:gxf.bzl", "nv_gxf_cc_library")
        '''

    def test_has_copyright(self):
        self.assertTrue(core.has_copyright(self.correct_1))
        self.assertTrue(core.has_copyright(self.correct_2))
        self.assertTrue(core.has_copyright(self.correct_3))
        self.assertTrue(core.has_copyright(self.correct_4))
        self.assertTrue(core.has_copyright(self.correct_5))
        self.assertTrue(core.has_copyright(self.wrong_1))
        self.assertTrue(core.has_copyright(self.wrong_2))
        self.assertTrue(core.has_copyright(self.wrong_3))
        self.assertTrue(core.has_copyright(self.wrong_4))
        self.assertTrue(core.has_copyright(self.wrong_5))
        self.assertFalse(core.has_copyright(self.wrong_6))

    def test_has_correct_copyright_year(self):
        self.assertTrue(core.has_correct_copyright_year(self.correct_1, TARGET_YEAR))
        self.assertTrue(core.has_correct_copyright_year(self.correct_2, TARGET_YEAR))
        self.assertTrue(core.has_correct_copyright_year(self.correct_3, TARGET_YEAR))
        self.assertTrue(core.has_correct_copyright_year(self.correct_4, TARGET_YEAR))
        self.assertTrue(core.has_correct_copyright_year(self.correct_5, TARGET_YEAR))
        self.assertFalse(core.has_correct_copyright_year(self.wrong_1, TARGET_YEAR))
        self.assertFalse(core.has_correct_copyright_year(self.wrong_2, TARGET_YEAR))
        self.assertFalse(core.has_correct_copyright_year(self.wrong_3, TARGET_YEAR))
        self.assertFalse(core.has_correct_copyright_year(self.wrong_4, TARGET_YEAR))
        self.assertFalse(core.has_correct_copyright_year(self.wrong_5, TARGET_YEAR))
        self.assertFalse(core.has_correct_copyright_year(self.wrong_6, TARGET_YEAR))

    def test_fix_copyright_year(self):

        def assert_fix(wrong, correct):
            fixed = core.fix_copyright_year(wrong, TARGET_YEAR)
            self.assertEqual(fixed, correct)

        assert_fix(self.wrong_1, self.correct_2)
        assert_fix(self.wrong_2, self.correct_2)
        assert_fix(self.wrong_3, self.correct_4)
        assert_fix(self.wrong_4, self.correct_4)
        assert_fix(self.wrong_5, self.correct_5)

        assert_fix(self.correct_2, self.correct_2)
        assert_fix(self.correct_4, self.correct_4)
        assert_fix(self.correct_5, self.correct_5)

    def test_check_header(self):
        fix = False
        self.assertTrue(check_header_helper(self.correct_1, TARGET_YEAR, fix))
        self.assertTrue(check_header_helper(self.correct_2, TARGET_YEAR, fix))
        self.assertTrue(check_header_helper(self.correct_3, TARGET_YEAR, fix))
        self.assertTrue(check_header_helper(self.correct_4, TARGET_YEAR, fix))

        self.assertFalse(check_header_helper(self.wrong_1, TARGET_YEAR, fix))
        self.assertFalse(check_header_helper(self.wrong_2, TARGET_YEAR, fix))
        self.assertFalse(check_header_helper(self.wrong_3, TARGET_YEAR, fix))
        self.assertFalse(check_header_helper(self.wrong_4, TARGET_YEAR, fix))


if __name__ == '__main__':
    unittest.main()
