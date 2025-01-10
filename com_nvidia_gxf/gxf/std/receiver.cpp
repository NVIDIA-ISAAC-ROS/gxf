/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/receiver.hpp"

namespace nvidia {
namespace gxf {

Expected<Entity> Receiver::receive() {
  gxf_uid_t uid;
  const gxf_result_t code = receive_abi(&uid);
  if (code == GXF_SUCCESS) {
    for (const Handle<Transmitter> transmitter : connected_transmitters_) {
      GXF_LOG_VERBOSE("Notifying upstream transmitter eid '%ld'.", transmitter->eid());
      GxfEntityNotifyEventType(context(), transmitter->eid(), GXF_EVENT_MESSAGE_SYNC);
    }
    return Entity::Own(context(), uid);
  } else {
    return Unexpected{code};
  }
}

size_t Receiver::back_size() {
  return back_size_abi();
}

Expected<void> Receiver::sync() {
  return ExpectedOrCode(sync_abi());
}

Expected<void> Receiver::sync_io() {
  return ExpectedOrCode(sync_io_abi());
}

Expected<void> Receiver::wait() {
  return ExpectedOrCode(wait_abi());
}

Expected<void> Receiver::setTransmitter(Handle<Transmitter> tx) {
  connected_transmitters_.insert(tx);
  return Success;
}

Expected<Entity> Receiver::peekBack(int32_t index) {
  gxf_uid_t uid;
  const gxf_result_t code = peek_back_abi(&uid, index);
  if (code == GXF_SUCCESS) {
    return Entity::Shared(context(), uid);
  } else {
    return Unexpected{code};
  }
}

}  // namespace gxf
}  // namespace nvidia
