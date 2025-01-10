/*
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gxf/multimedia/camera.hpp"


PYBIND11_MODULE(camera_pybind, m) {
  // Bind DistortionType enum
  pybind11::enum_<nvidia::gxf::DistortionType>(m, "DistortionType")
      .value("Perspective", nvidia::gxf::DistortionType::Perspective)
      .value("Brown", nvidia::gxf::DistortionType::Brown)
      .value("Polynomial", nvidia::gxf::DistortionType::Polynomial)
      .value("FisheyeEquidistant", nvidia::gxf::DistortionType::FisheyeEquidistant)
      .value("FisheyeEquisolid", nvidia::gxf::DistortionType::FisheyeEquisolid)
      .value("FisheyeOrthoGraphic", nvidia::gxf::DistortionType::FisheyeOrthoGraphic)
      .value("FisheyeStereographic", nvidia::gxf::DistortionType::FisheyeStereographic);

  // Bind Vector2 template
  pybind11::class_<nvidia::gxf::Vector2<uint32_t>>(m, "Vector2u")
      .def(pybind11::init<uint32_t, uint32_t>())
      .def_readwrite("x", &nvidia::gxf::Vector2<uint32_t>::x)
      .def_readwrite("y", &nvidia::gxf::Vector2<uint32_t>::y);

  pybind11::class_<nvidia::gxf::Vector2<float>>(m, "Vector2f")
      .def(pybind11::init<float, float>())
      .def_readwrite("x", &nvidia::gxf::Vector2<float>::x)
      .def_readwrite("y", &nvidia::gxf::Vector2<float>::y);

  pybind11::class_<nvidia::gxf::Vector2<double>>(m, "Vector2d")
      .def(pybind11::init<double, double>())
      .def_readwrite("x", &nvidia::gxf::Vector2<double>::x)
      .def_readwrite("y", &nvidia::gxf::Vector2<double>::y);

  // Bind CameraModel over CameraModelBase template with float
  pybind11::class_<nvidia::gxf::CameraModelBase<float>>(m, "CameraModel")
      .def(pybind11::init<>())
      .def_readwrite("dimensions", &nvidia::gxf::CameraModelBase<float>::dimensions)
      .def_readwrite("focal_length", &nvidia::gxf::CameraModelBase<float>::focal_length)
      .def_readwrite("principal_point", &nvidia::gxf::CameraModelBase<float>::principal_point)
      .def_readwrite("skew_value", &nvidia::gxf::CameraModelBase<float>::skew_value)
      .def_readwrite("distortion_type", &nvidia::gxf::CameraModelBase<float>::distortion_type)
      .def_readwrite("distortion_coefficients", &nvidia::gxf::CameraModelBase<float>::distortion_coefficients);

  // Bind Pose3D over Pose3DBase template with float
  pybind11::class_<nvidia::gxf::Pose3DBase<float>>(m, "Pose3D")
      .def(pybind11::init<>())
      .def_readwrite("rotation", &nvidia::gxf::Pose3DBase<float>::rotation)
      .def_readwrite("translation", &nvidia::gxf::Pose3DBase<float>::translation);
}
