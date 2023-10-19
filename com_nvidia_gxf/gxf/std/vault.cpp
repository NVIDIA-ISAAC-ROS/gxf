/*
Copyright (c) 2021,2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "gxf/std/vault.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t Vault::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &=
      registrar->parameter(source_, "source", "Source",
                           "Receiver from which messages are taken and transferred to the vault.");
  result &= registrar->parameter(
      max_waiting_count_, "max_waiting_count", "Maximum waiting count",
      "The maximum number of waiting messages. If exceeded the codelet will stop pulling messages "
      "out of the input queue.");
  result &= registrar->parameter(drop_waiting_, "drop_waiting", "Drop waiting",
                                 "If too many messages are waiting the oldest ones are dropped.");
  result &= registrar->parameter(callback_address_, "callback_address",
                                 "Callback address", "", 0l);
  result &= registrar->parameter(enable_callback_, "enable_callback", "Enable Callback",
      "Enable Callback", false);
  return ToResultCode(result);
}

gxf_result_t Vault::initialize() {
  alive_ = true;
  return GXF_SUCCESS;
}

gxf_result_t Vault::start() {
  int64_t callback_address = callback_address_;
  if (callback_address != 0) {
    GXF_LOG_DEBUG("Setting callback address from int64_t [%05zu]", callback_address);
    callback_.reset(new CallbackType([callback_address]() {
      if (callback_address) {
        // Dereference the int64_t pointer and call the associated function
        CallbackType* callback_type_ptr = reinterpret_cast<CallbackType*>(callback_address);
        (*callback_type_ptr)();
      } else {
        GXF_LOG_WARNING("Calling invalid callback, because of invalid callback address");
      }
    }));
  }
  return GXF_SUCCESS;
}

gxf_result_t Vault::tick() {
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // If too many messages are waiting we make room for new ones
    if (drop_waiting_ && entities_waiting_.size() + source_->size() > max_waiting_count_) {
      const size_t rmv_count = entities_waiting_.size() + source_->size() - max_waiting_count_;
      entities_waiting_.erase(entities_waiting_.begin(), entities_waiting_.begin() + rmv_count);
    }

    // Move incoming messages to the queue of waiting messages.
    while (entities_waiting_.size() < max_waiting_count_) {
      auto message = source_->receive();
      if (!message) {
        if (enable_callback_ == true) {
          // Break to continue handling any messages left in entities_waiting_
          // changes cv notification behavior
          break;
        } else {
          return GXF_SUCCESS;
        }
      }

      entities_waiting_.emplace_back(std::move(message.value()));
    }
  }

  if (enable_callback_ == true) {
    if (entities_waiting_.size() == 0) {
      return GXF_SUCCESS;
    }

    if (callback_) {
      GXF_LOG_DEBUG("Invoking callback function");
      (*callback_)();
    } else {
      GXF_LOG_DEBUG("Callback enabled, but forgot to set callback function");
    }
  }

  // Notify potentially waiting storage requests about the new messages.
  condition_variable_.notify_one();

  return GXF_SUCCESS;
}

gxf_result_t Vault::stop() {
  return GXF_SUCCESS;
}

gxf_result_t Vault::deinitialize() {
  alive_ = false;

  // Terminate all blocking requests.
  condition_variable_.notify_one();

  return GXF_SUCCESS;
}


gxf_result_t Vault::setCallback(CallbackType callback) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (callback_ != nullptr) {
    GXF_LOG_WARNING("Attempting to reset callback function pointer from %p to %p",
                    callback_.get(), &callback);
  }
  callback_ = std::make_unique<CallbackType>(std::move(callback));
  return GXF_SUCCESS;
}

std::vector<gxf_uid_t> Vault::storeBlocking(size_t count) {
  // Wait for the desired number of messages
  std::unique_lock<std::mutex> lock(mutex_);
  condition_variable_.wait(lock,
                           [this, count] { return !alive_ || entities_waiting_.size() >= count; });

  return storeImpl(count);
}

std::vector<gxf_uid_t> Vault::storeBlockingFor(size_t count, int64_t duration_ns) {
  if (duration_ns < 0.0) {
    return {};
  }
  // Wait for the desired number of messages with a target duration for time out
  std::unique_lock<std::mutex> lock(mutex_);
  condition_variable_.wait_for(lock, std::chrono::duration<int64_t, std::nano>(duration_ns),
                           [this, count] { return !alive_ || entities_waiting_.size() >= count; });

  return storeImpl(count);
}

std::vector<gxf_uid_t> Vault::store(size_t max_count) {
  std::unique_lock<std::mutex> lock(mutex_);

  return storeImpl(max_count);
}

std::vector<gxf_uid_t> Vault::storeImpl(size_t max_count) {
  // In case the codelet is shutting down we do not have anything to return
  // TODO(dweikersdorf) Additional checks might be necessary to make sure that the codelet does no
  //                    finish deinitialization until all storeBlocking calls have returned.
  std::vector<gxf_uid_t> result;
  if (!alive_) {
    return result;
  }

  // Store the UIDs of the messages
  size_t count = std::min(max_count, entities_waiting_.size());
  result.reserve(count);
  for (size_t i = 0; i < count; i++) {
    result.emplace_back(entities_waiting_[i].eid());
  }

  // Move the messages from the queue of waiting messages to the vault.
  const auto it1 = entities_waiting_.begin();
  const auto it2 = entities_waiting_.begin() + count;
  entities_in_vault_.insert(entities_in_vault_.end(), std::make_move_iterator(it1),
                            std::make_move_iterator(it2));
  entities_waiting_.erase(it1, it2);

  return result;
}

void Vault::free(const std::vector<gxf_uid_t>& entities) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Find each message in the vault and remove it.
  for (gxf_uid_t uid : entities) {
    auto it = std::remove_if(entities_in_vault_.begin(), entities_in_vault_.end(),
                             [uid](const Entity& entity) { return entity.eid() == uid; });
    entities_in_vault_.erase(it, entities_in_vault_.end());
  }
}

}  // namespace gxf
}  // namespace nvidia
