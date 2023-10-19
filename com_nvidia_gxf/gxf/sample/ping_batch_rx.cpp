/*
Copyright (c) 2023 NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/sample/ping_batch_rx.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t PingBatchRx::start() {
  return GXF_SUCCESS;
}

gxf_result_t PingBatchRx::tick() {
  for (int64_t i = 0; i < batch_size_; i++) {
    auto message = signal_->receive();
    GXF_LOG_DEBUG("Message received in ping batch rx");
    if (assert_full_batch_ && (!message || message.value().is_null())) {
      return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t PingBatchRx::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(signal_, "signal");
  result &= registrar->parameter(batch_size_, "batch_size");
  result &= registrar->parameter(assert_full_batch_, "assert_full_batch", "Assert Full Batch",
                                 "Assert if the batch is not fully populated.", true);
  return ToResultCode(result);
}

}  // namespace gxf
}  // namespace nvidia
