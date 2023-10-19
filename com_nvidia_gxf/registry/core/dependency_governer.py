# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
""" Dependency Governor
"""

import json
import math
from math import ceil, log
from os import close
from typing import List
from packaging import version

from registry.core.extension import Extension
from registry.core.utils import ExtensionRecord, TargetConfig, ComputeConfig, target_to_str
import registry.core.logger as log

logger = log.get_logger("Registry")
class DependencyGovernor:
    """ A utility class used to manage extension dependencies for new extension registration
        and graph install operations
    """
    def __init__(self, database):

        self._db = database

    def find_all_dependencies(self, deps: List[ExtensionRecord]):
        """ Find all direct and in-direct dependencies for a list of Dependencies

        Args:
            deps List[ExtensionRecord]: List of ExtensionRecord for which all the dependencies need to be queried
        Returns:
            List[ExtensionRecord]: List of ExtensionRecord
        """

        deps_graph = self._create_dependency_graph(deps)

        if deps_graph is None:
            logger.error("Failed to find dependencies")
            return None

        if not deps_graph:
            logger.debug(f"No dependencies found for {deps}")
            return []

        if not self.graph_version_check(deps_graph):
            return None

        logger.debug("Complete dependency graph " +
                           str(self._visualize_dependency_graph(deps_graph)))

        global_visited = set()
        dependent_exts = []
        for dep in deps:
            exts, ext_visited = self._graph_traverse(deps_graph, dep, global_visited)
            global_visited.union(ext_visited)
            [dependent_exts.append(e) for e in exts if e not in dependent_exts]

        logger.debug("Dependent extensions required " +
                           str(self._visualize_extension_list(dependent_exts)))

        uuids = set()
        for ext in dependent_exts:
            if ext.uuid not in uuids:
                uuids.add(ext.uuid)
            else:
                logger.error(f"Multiple versions of {ext.name} extension found in dependency tree")
                logger.error("Complete dependency graph")
                logger.error(self._visualize_dependency_graph(deps_graph))
                return None

        unique_exts = self.version_update(dependent_exts)
        return unique_exts

    def _create_dependency_graph(self, nodes, deps_graph={}):
        for node in nodes:
            deps = self._db.get_dependencies(node)
            if deps is None:
                return None
            deps_graph[node] = deps

        # recursively look for indirect dependencies
        indirect_deps = self._graph_has_indirect_deps(deps_graph)
        while indirect_deps:
            r_deps_graph = self._create_dependency_graph(indirect_deps, deps_graph)
            if r_deps_graph is None:
                return None
            else:
                deps_graph.update(r_deps_graph)
            indirect_deps = self._graph_has_indirect_deps(deps_graph)

        return deps_graph

    def _graph_has_indirect_deps(self, deps_graph):
        """ Parses dependency graph to search for indirect dependencies

        Args:
            deps_graph (Dict{ExtensionRecord:[ExtensionRecord]}): dependency graph of extensions

        Returns:
            List[ExtensionRecord]: List of ExtensionRecord corresponding to indirect deps
        """

        result = []
        for deps in deps_graph.values():
            for dep in deps:
                if dep not in deps_graph.keys():
                    result.append(dep)
        indirect_deps = []
        [indirect_deps.append(dep) for dep in result if dep not in indirect_deps]

        return indirect_deps

    def _graph_traverse(self, deps_graph, node, visited):
        """ Depth first graph traversal to find all the dependencies in
            "leaf first" order

        Args:
            deps_graph (Dict{ExtensionRecord:[ExtensionRecord]}): dependency graph of extensions
            node (ExtensionRecord): starting extension for graph traversal
            visited (set): set of extensions that have already been visited in the graph

        Returns:
            List[ExtensionRecord]: List of ExtensionRecord extensions in order to be loaded
            set: Updated set of extensions that have been visited in the graph
        """

        manifest = []
        if node not in visited:
            for dep in deps_graph[node]:
                dep_manifest, dep_visited = self._graph_traverse(deps_graph, dep, visited)
                # Extension (and its deps) is already visited
                if dep_manifest == []:
                    continue
                # Extension either has dependencies and they have already been visited,
                # OR Extension has no dependencies, and manifest is just the extension itself
                # OR Extension had some unvisited deps along with itself
                # In all 3 cases, append the newly visited extension at the end
                else:
                    manifest += dep_manifest
                visited.union(dep_visited)
            manifest.append(node)
            visited.add(node)

        return manifest, visited

    def _visualize_dependency_graph(self, deps_graph):
        viz_graph = {}
        for ext in deps_graph:
            cur_ext_deps = []
            for dep in deps_graph[ext]:
                cur_ext_deps.append(f"{dep.name}:{dep.version}")
            viz_graph[f"{ext.name}:{ext.version}"] = cur_ext_deps

        return json.dumps(viz_graph, indent=2)

    def _visualize_extension_list(self, extension_list):
        extensions = []
        for ext in extension_list:
            extensions.append(f"{ext.name}:{ext.version}")
        return extensions

    def graph_version_check(self, deps_graph):
        """ Checks if an extension in a dependency graph has multiple versions corresponding
            to different major versions

        Args:
            deps_graph (Dict{ExtensionRecord:[ExtensionRecord]}): dependency graph of extensions

        Returns:
            bool: "False" if an extension in a dependency graph has multiple versions with different
                  major versions, "True" otherwise.
        """
        version_registry = {}

        def _add_to_registry(ext):
            if ext.uuid not in version_registry.keys():
                version_registry[ext.uuid] = ext.version
            elif version_registry[ext.uuid] != ext.version:
                reg_ver = version.parse(version_registry[ext.uuid])
                cur_ver = version.parse(ext.version)
                if cur_ver.major != reg_ver.major:
                    logger.error(f"Conflicting versions found for {ext.name}")
                    logger.error(
                        f"Version 1: {ext.version} Version 2: {version_registry[ext.uuid]}")
                    logger.error(
                        f"Loading multiple versions of the same extension is not supported")
                    return False
            return True

        for ext, deps in deps_graph.items():
            if not _add_to_registry(ext):
                return False
            for dep in deps:
                if not _add_to_registry(dep):
                    return False

        return True

    def version_update(self, exts: List[ExtensionRecord]):
        """returns a unique list of extensions with the version updated from the database

        Args:
            exts (List[ExtensionRecord]): list of extensions

        Returns:
            List[ExtensionRecord]: unique list of extensions with the major version updated
        """
        unique_exts = {}
        for ext in exts:
            if ext.uuid in unique_exts.keys():
                continue
            else:
                unique_exts[ext.uuid] = self._db.get_version_update(ext)

        return list(unique_exts.values())

    def get_best_variants(self, exts: List[ExtensionRecord], target_cfg: TargetConfig):
        """ Queries for the closest matching variant for all the extension given the target_cfg

        Args:
            exts (List[ExtensionRecord]): list of extension inputs
            target_cfg (TargetConfig): query target configuration to be matched
        """

        result_variants = {}
        for ext in exts:
            repo_name = self._db.get_extension_source_repo(ext)
            variants = self._db.get_variants(ext.uuid, ext.version, repo_name)
            targets = [var.target for var in variants]
            ext_name = self._db.get_extension_name(ext.uuid)

            closest_variant = None
            match_score = float('-inf')
            for var in targets:
                if var.platform != target_cfg.platform:
                    continue

                score = self._compute_dependency_match(target_cfg.compute, var.compute)
                if score >= match_score:
                    match_score = score
                    closest_variant = var
                    logger.debug(f"Found a close match {var}")

            if not match_score >= 0:
                logger.error(
                    f"No matching variant found for extension: {ext_name} version: {ext.version}")
                logger.error(f"Requested target configuration {target_to_str(target_cfg)}")
                logger.error(
                    f"Extension variants found {[target_to_str(v) for v in targets]}")
                return None

            result_variants[ext] = closest_variant

        summary_str = f"\nRequested target configuration: \n  {target_to_str(target_cfg)}\n\n"
        summary_str += "Selected extension variants: \n"
        for ext, var in result_variants.items():
            ext_name = self._db.get_extension_name(ext.uuid)
            summary_str += f"{ext_name} \n  Version: {ext.version} "
            summary_str += f"Variant: {target_to_str(var)}\n"

        logger.info(summary_str)
        return result_variants

    def _compute_dependency_match(self, query: ComputeConfig, value: ComputeConfig):
        """ Compute the similarity scores between two compute configs

        Args:
            query (ComputeConfig): query compute config
            value (ComputeConfig): value compute config
        """

        score = 0
        if query.cuda:
            score += self._version_compare(query.cuda, value.cuda)
        if query.cudnn:
            score += self._version_compare(query.cudnn, value.cudnn)
        if query.tensorrt:
            score += self._version_compare(query.tensorrt, value.tensorrt)
        if query.deepstream:
            score += self._version_compare(query.deepstream, value.deepstream)
        if query.triton:
            score += self._version_compare(query.triton, value.triton)
        if query.vpi:
            score += self._version_compare(query.vpi, value.vpi)

        return score

    def _version_compare(self, query: str, value: str):
        """ Compute the similarity score between two version strings

        Args:
            query (str): query version string
            value (str): value version string
        """
        # If variant does not depend on compute stack, rate this lowest
        if not value:
            return 0

        q_ver = version.parse(str(query))
        v_ver = version.parse(str(value))

        # If variant's major version does not match with target's major version,
        # set score to neg inf to ensure it's not used
        if q_ver.major != v_ver.major:
            return float('-inf')

        # Start with score 100 for matching major versions
        score = 100
        # if q_ver.major == v_ver.major:
        #    score += 100
        # else:
        #     m_score = 100 * (q_ver.major - v_ver.major) / (10**ceil(
        #         log(max(q_ver.major, v_ver.major), 10)))
        #     if m_score > 0:
        #         score += 100 - m_score
        #     else:
        #         score -= 100 + m_score

        if q_ver.minor == v_ver.minor:
            score += 10
        else:
            m_score = 10 * (q_ver.minor - v_ver.minor) / (10**math.ceil(
                math.log(max(q_ver.minor, v_ver.minor), 10)))
            if m_score > 0:
                score += 10 - m_score
            else:
                score -= 10 + m_score

        if q_ver.micro == v_ver.micro:
            score += 1
        else:
            m_score = 1 * (q_ver.micro - v_ver.micro) / (10**math.ceil(
                math.log(max(q_ver.micro, v_ver.micro), 10)))
            if m_score > 0:
                score += 1 - m_score
            else:
                score -= 1 + m_score

        return score
