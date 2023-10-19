/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef NVIDIA_GXF_EVENT_LIST_HPP_
#define NVIDIA_GXF_EVENT_LIST_HPP_

#include <algorithm>
#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"

namespace nvidia {
namespace gxf {

// A simple thread safe list implementation to store and retrive events
template <typename T>
class EventList {
 public:
  EventList() = default;
  EventList(const EventList<T>&) = delete;
  EventList& operator=(const EventList<T>&) = delete;

  // Adds an item on to the event list
  void pushEvent(T item) {
    std::lock_guard<std::mutex> lock(list_mutex_);
    list_.push_back(item);
  }

  // Removes an event from the list if it exists and returns true,
  // returns false otherwise
  void removeEvent(T item) {
    std::lock_guard<std::mutex> lock(list_mutex_);

    for (auto it = list_.begin(); it != list_.end();) {
      if (*it == item) {
        // remove all instances in case of multiple event notifications
        // for the same entity
        it = list_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // pops the first element in the list
  Expected<T> popEvent() {
    std::lock_guard<std::mutex> lock(list_mutex_);
    if (list_.empty()) { return Unexpected{GXF_FAILURE}; }

    T event = list_.front();
    list_.pop_front();
    return event;
  }

  // exports a copy of the events in the list
  std::list<T> exportList() const {
    std::lock_guard<std::mutex> lock(list_mutex_);
    return list_;
  }

  // checks if the list is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(list_mutex_);
    return list_.empty();
  }

  // returns size of the event list
  size_t size() const {
    std::lock_guard<std::mutex> lock(list_mutex_);
    return list_.size();
  }

  // checks if the list has event
  bool hasEvent(T item) const {
    std::lock_guard<std::mutex> lock(list_mutex_);
    for (const auto& element : list_) {
      if (item == element) { return true; }
    }
    return false;
  }

  // clears the event list
  void clear() {
    std::lock_guard<std::mutex> lock(list_mutex_);
    list_.clear();
  }

  mutable std::mutex list_mutex_;
  std::list<T> list_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_EVENT_LIST_HPP_
