'''
 SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import argparse
import json
import multiprocessing
import os
import subprocess
import sys

"""
Run a single app as a subprocess and return its return code and logs. stdout and stderr logs
are combined into a single stream for simplicity.
"""
def run_single_app(app_target):
    print("================================================================================")
    print(f"RUNNING App {app_target}")
    print("================================================================================")
    sys.stdout.flush()

    # TODO Capture output logs in a buffer for debugging failed apps
    app_process = subprocess.Popen(app_target, shell=False)
                                #    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                #    text=True, universal_newlines=True)
    stdout, _ = app_process.communicate()
    app_process.wait()
    print("================================================================================")
    print(f"Finished App {app_target}")
    print("================================================================================")
    sys.stdout.flush()

    return (app_target, app_process.returncode, stdout)

"""
Run multiple apps in parallel and returns the count of failed apps.
"""
def run_multiple_apps(app_targets: list):

    process_map = {}
    failed_apps = {}

    pool = multiprocessing.Pool()
    results = pool.map(run_single_app, app_targets.keys())
    pool.close()
    pool.join()

    failed_apps = [(app, return_code, stdout) for app, return_code, stdout in results if return_code != 0]
    for app, _ , stdout in failed_apps:
        print("*******************************************************************************")
        print(f"App {app} FAILED with log : ")
        print(stdout)
        print("*******************************************************************************")

    return len(failed_apps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("app_targets",
                        help='JSON-formatted string containing commands to execute and their args',
                        type=str)

    args = parser.parse_args()
    apps = json.loads(args.app_targets)
    result = run_multiple_apps(apps)

    if result == 0:
        print("All apps executed successfully")

    print("Exiting.....")
    sys.exit(result)
