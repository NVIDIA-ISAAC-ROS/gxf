/*
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/cuda/cuda_stream_sync.hpp"

#include "gxf/cuda/cuda_common.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"

#include <string>

namespace nvidia {
namespace gxf {

gxf_result_t CudaStreamSync::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(rx_, "rx", "Receiver");
  result &= registrar->parameter(tx_, "tx", "Transmitter", "", Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t CudaStreamSync::tick() {
  if (!rx_.get()) {
    GXF_LOG_ERROR("rx of CudaStreamsync is not set");
    return GXF_PARAMETER_MANDATORY_NOT_SET;
  }
  // Read message from receiver
  const auto message = rx_->receive();
  if (!message) {
    return ToResultCode(message);
  }

  auto stream_ids = message->findAll<CudaStreamId>();
  if (!stream_ids) {
    return ToResultCode(stream_ids);
  }

  for (auto stream_id : stream_ids.value()) {
    if (!stream_id) {
      GXF_LOG_ERROR("Received null handle to a cuda stream id");
      return GXF_FAILURE;
    }
    if (stream_id->is_null() || stream_id.value()->stream_cid == kNullUid) {
      GXF_LOG_WARNING("CudaStreamSync received empty cudastreamid: %s,"
                      "skip and cointinue", stream_id->name());
      continue;
    }
    auto stream = Handle<CudaStream>::Create(context(), stream_id.value()->stream_cid);
    if (!stream) {
      GXF_LOG_ERROR("CudaStreamId(name: %s) convert to CudaStream failed.", stream_id->name());
      return GXF_ENTITY_COMPONENT_NOT_FOUND;
    }
    GXF_ASSERT(!stream->is_null(), "stream should not be null");
    auto ret = stream.value()->syncStream();
    if (!ret) {
        GXF_LOG_ERROR("CudaStreamSync tick and sync stream: %s failed.", stream->name());
        return ToResultCode(ret);
    }
    GXF_LOG_DEBUG("CudaStreamSync tick on stream: %s", stream->name());
  }

  // Publish message
  auto tx = tx_.try_get();
  if (tx) {
    return ToResultCode(tx.value()->publish(message.value()));
  } else {
    return GXF_SUCCESS;
  }
}

}  // namespace gxf
}  // namespace nvidia
