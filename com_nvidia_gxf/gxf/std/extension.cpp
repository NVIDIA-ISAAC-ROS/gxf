/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/extension.hpp"

namespace nvidia {
namespace gxf {

Expected<void> Extension::registerComponents(gxf_context_t context) {
  return ExpectedOrCode(registerComponents_abi(context));
}

Expected<void> Extension::getComponentTypes(gxf_tid_t* pointer, size_t* size) {
  return ExpectedOrCode(getComponentTypes_abi(pointer, size));
}

Expected<void> Extension::setInfo(gxf_tid_t tid, const char* name, const char* desc,
                                   const char* author, const char* version, const char* license) {
  return ExpectedOrCode(setInfo_abi(tid, name, desc, author, version, license));
}

Expected<void> Extension::setDisplayInfo(const char* display_name, const char* category,
                                         const char* brief) {
  return ExpectedOrCode(setDisplayInfo_abi(display_name, category, brief));
}

Expected<void> Extension::checkInfo() {
  return ExpectedOrCode(checkInfo_abi());
}

Expected<void> Extension::getInfo(gxf_extension_info_t* info) {
  return ExpectedOrCode(getInfo_abi(info));
}

Expected<void> Extension::getComponentInfo(const gxf_tid_t tid, gxf_component_info_t* info) {
  return ExpectedOrCode(getComponentInfo_abi(tid, info));
}

Expected<void> Extension::getParameterInfo(gxf_context_t context, const gxf_tid_t cid,
                                           const char* key, gxf_parameter_info_t* info) {
  return ExpectedOrCode(getParameterInfo_abi(context, cid, key, info));
}

}  // namespace gxf
}  // namespace nvidia
