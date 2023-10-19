#####################################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

import os

PACKAGES = []
with open(f"{os.path.realpath(os.path.dirname(__file__))}/../packages.txt","r") as f:
    PACKAGES = f.read().splitlines()

def filter_packages(terms,num_builds_to_preserve):
    do_not_purge = []
    for package_type in PACKAGES:
        corresponding_packages = list(filter((lambda x: x["name"].count(package_type) or x["type"] != "file"), terms))
        corresponding_packages = corresponding_packages[-num_builds_to_preserve:]
        do_not_purge += corresponding_packages

    return do_not_purge

def purgelist(artifactory):

    purgable = []

    # purge anything other than last 30 builds in sw-isaac-gxf-generic-local/nightly/master
    num_nightly_builds = 30
    compute_nightly_master_terms = [
        {"path": {"$match": "nightly/master"}}
    ]
    nightly_master_terms = artifactory.filter(terms=compute_nightly_master_terms, depth=None, item_type="any")

    purgable += [i for i in nightly_master_terms if i not in filter_packages(nightly_master_terms,num_nightly_builds)]

    # purge anything other than last 30 builds in sw-isaac-gxf-generic-local/nightly/release-
        # 2.3
        # 2.3.1
        # 2.3.2
        # 2.3.3
        # 2.4
        # 2.4.1
        # 2.4.2
        # 2.4.3
    nightly_release_versions = ["2.3","2.3.1","2.3.2","2.3.3","2.4","2.4.1","2.4.2","2.4.3",]
    num_release_builds = 30
    for version in nightly_release_versions:
        compute_nightly_release_terms = [
            {"path": {"$match": "nightly/release-{_version}".format(_version=version)}}
        ]
        nightly_release_terms = artifactory.filter(terms=compute_nightly_release_terms, depth=None, item_type="any")

        purgable += [i for i in nightly_release_terms if i not in filter_packages(nightly_release_terms,num_release_builds)]

    # purge anything created more than 7 nightly runs ago in sw-isaac-gxf-generic-local/nightly/pipeline-testing
    num_nightly_test_builds = 7
    compute_nightly_test_terms = [
        {"path": {"$match": "nightly/pipeline-testing"}}
    ]
    nightly_test_terms = artifactory.filter(terms=compute_nightly_test_terms, depth=None)

    purgable += [i for i in nightly_test_terms if i not in filter_packages(nightly_test_terms,num_nightly_test_builds)]

    # trim list of items to be purged with the rule:
    # if name is like xxx_public.tar
    # do not purge anything that starts with xxx_
    do_not_purge = [will_purge["name"].partition("public.tar")[0]
        for will_purge in purgable
            if will_purge["name"].endswith("public.tar")]
    purgable = [will_purge
        for will_purge in purgable
            if not any(will_purge["name"].startswith(no_purge)
                for no_purge in do_not_purge)]

    return purgable
