/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <iostream>
#include <string>

#include "common/logger.hpp"
#include "gxf/core/gxf.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

#include "pygxf.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pygxf, m) {
  m.doc() = "pybind11 bindings for gxf core";

  m.def("gxf_result_str", &gxf_result_str, "Gives a string describing a result",
        py::arg("result"));

  m.def("gxf_context_create", &gxf_context_create,
        py::return_value_policy::reference,
        "Creates a GXF context which is required for all other GXF operations");

  // FIXME: Add destructor to capsule to use garbage collector
  m.def("gxf_context_destroy", &gxf_context_destroy,
        "Destroys a GXF context. This function blocks until the application is "
        "destroyed gracefully.",
        py::arg("context"));

  m.def("gxf_register_component", &gxf_register_component,
        "Registers a component type which can be used for GXF core.",
        py::arg("context"), py::arg("tid"), py::arg("name"),
        py::arg("base_name"));

  m.def("gxf_load_ext", &gxf_load_ext, "Loads an extension library.",
        py::arg("context"), py::arg("filename"));

  m.def("gxf_load_ext_manifest", &gxf_load_ext_manifest,
        "Loads multiple extensions as specified in the manifest file. The "
        "manifest file is a YAML file which lists extensions to be loaded.",
        py::arg("context"), py::arg("filename"));

  m.def("gxf_load_extensions", &gxf_load_extensions,
        "Loads multiple extensions as specified in list.",
        py::arg("context"), py::arg("filenames"));

  m.def("gxf_load_extension_metadata", &gxf_load_extension_metadata,
        "Loads multiple extension metadata files as specified in list.",
        py::arg("context"), py::arg("filenames"));

  m.def("gxf_load_graph_file", &gxf_load_graph_file,
        "Loads a list of entities from a YAML file.", py::arg("context"),
        py::arg("filename"));

  m.def("gxf_graph_run", &gxf_graph_run,
        "Runs all System components and waits for their completion",
        py::arg("context"));

  m.def("gxf_graph_activate", &gxf_graph_activate,
        "Activate all System components", py::arg("context"));

  m.def("gxf_graph_deactivate", &gxf_graph_deactivate,
        "Deactivate all System components", py::arg("context"));

  m.def("gxf_graph_run_async", &gxf_graph_run_async,
        "Starts the execution of the graph asynchronously", py::arg("context"));

  m.def("gxf_graph_interrupt", &gxf_graph_interrupt,
        "Interrupt the execution of the graph", py::arg("context"));

  m.def("gxf_graph_wait", &gxf_graph_wait,
        "Waits for the graph to complete execution", py::arg("context"));

  // -- Query ------------------------------

  py::class_<gxf_runtime_info>(m, "gxf_runtime_info").def(py::init<>());

  m.def("get_runtime_version", &get_runtime_version,
        "Gets Meta Data about the GXF Runtime", py::arg("context"));

  m.def("get_ext_list", &get_ext_list,
        "Gets list of extension uuid's (eid) from Gxf Runtime",
        py::arg("context"));

  m.def("get_ext_info", &get_ext_info,
        "Gets description and info in loaded Extension", py::arg("context"),
        py::arg("eid"));

  m.def("get_comp_list", &get_comp_list,
        "Gets list of component uuid's (cid) from Gxf Runtime for an extension",
        py::arg("context"), py::arg("eid"));

  m.def("get_comp_info", &get_comp_info,
        "Gets description and info in loaded Extension", py::arg("context"),
        py::arg("cid"));

  m.def("get_param_list", &get_param_list,
        "Gets list of parameter keys from Gxf Runtime for a component",
        py::arg("context"), py::arg("cid"));

  m.def("get_param_info", &get_param_info,
        "Gets description and info in loaded Extension", py::arg("context"),
        py::arg("cid"), py::arg("key"));

}  // PYBIND11_MODULE
