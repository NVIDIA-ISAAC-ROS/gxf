/*
Copyright (c) 2020,2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <memory>
#include <vector>
#include "gxf/core/gxf.h"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/resources.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/system.hpp"

namespace nvidia {
namespace gxf {

// A simple poll-based single-threaded scheduler which executes codelets.
class Scheduler : public System {
 public:
  virtual gxf_result_t prepare_abi(EntityExecutor* executor) = 0;
};

}  // namespace gxf
}  // namespace nvidia
