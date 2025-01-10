/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <vector>

#include "gxf/app/graph_entity.hpp"

namespace nvidia {
namespace gxf {

namespace {
static const gxf_tid_t TransmitterTID{0xc30cc60f0db2409d, 0x92b6b2db92e02cce};
static const gxf_tid_t ReceiverTID{0xa47d2f62245f40fc, 0x90b75dc78ff2437e};
}  // namespace

Expected<std::string> GraphEntity::formatTxName(const char* tx_name) {
  auto tx_info = UNWRAP_OR_RETURN(codelet_->getParameterInfo(tx_name));
  if (tx_info.type != GXF_PARAMETER_TYPE_HANDLE || tx_info.handle_tid != TransmitterTID) {
    GXF_LOG_ERROR("Tx name [%s] in entity [%s] does not correspond to a transmitter parameter",
                  tx_name, name());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // scalar output port parameter
  if (tx_info.rank == 0) { return tx_name; }

  // vector / array output port parameter
  auto param = codelet_->wrapParameter(tx_name);
  if (!param && param.error() == GXF_PARAMETER_NOT_INITIALIZED) {
    return std::string(tx_name) + "_0";
  }

  return std::string(tx_name) + "_" + std::to_string(param.value().size());
}

Expected<std::string> GraphEntity::formatRxName(const char* rx_name) {
  auto rx_info = UNWRAP_OR_RETURN(codelet_->getParameterInfo(rx_name));
  if (rx_info.type != GXF_PARAMETER_TYPE_HANDLE || rx_info.handle_tid != ReceiverTID) {
    GXF_LOG_ERROR("Rx name [%s] in entity [%s] does not correspond to a receiver parameter",
                  rx_name, name());
    return Unexpected{GXF_ARGUMENT_INVALID};
  }

  // scalar input port parameter
  if (rx_info.rank == 0) { return rx_name; }

  // vector / array input port parameter
  auto param = codelet_->wrapParameter(rx_name);
  if (!param && param.error() == GXF_PARAMETER_NOT_INITIALIZED) {
    return std::string(rx_name) + "_0";
  }

  return std::string(rx_name) + "_" + std::to_string(param.value().size());
}

Expected<void> GraphEntity::updatePort(const char* key, std::string value) {
  GXF_LOG_DEBUG("Updating port parameter [%s] in codelet [%s] with value [%s]", key, name(),
                value.c_str());
  auto info = UNWRAP_OR_RETURN(codelet_->getParameterInfo(key));
  auto param = codelet_->wrapParameter(key);
  YAML::Node node;

  if (!param && param.error() == GXF_PARAMETER_NOT_INITIALIZED) {
    if (info.rank == 0) {  // scalar
      node = YAML::Node(value);
    } else if (info.rank == 1 && info.shape[0] == -1) {  // vector<T>
      node = YAML::Node(YAML::NodeType::Sequence);
      node.push_back(value);
    } else if (info.rank == 1 && info.shape[0] > 0) {  // array<T, N>
      node = YAML::Node(YAML::NodeType::Sequence);
      node.push_back(value);
    } else {
      GXF_LOG_ERROR("Invalid parameter [%s] rank / shape. Cannot be updated", key);
      return Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
    }
  } else if (param) {
    if (info.rank == 0) {  // scalar
      GXF_LOG_ERROR("Scalar parameter [%s] has already been set", key);
      return Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
    } else if (info.rank == 1 && info.shape[0] == -1) {  // vector<T>
      node = param.value();
      node.push_back(value);
    } else if (info.rank == 1 && info.shape[0] > 0) {  // array<T, N>
      node = param.value();
      node.push_back(value);
    } else {
      GXF_LOG_ERROR("Invalid parameter [%s] rank / shape. Cannot be updated", key);
      return Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
    }
  } else {
    GXF_LOG_ERROR("Failed to update codelet parameter [%s] with value [%s]", key, value.c_str());
    return Unexpected{GXF_FAILURE};
  }

  return codelet_->parseParameter(key, node);
}

Handle<Component> GraphEntity::get(const char* type_name, const char* name) const {
  gxf_tid_t tid;
  auto null_handle = Handle<Component>::Null();
  const auto result = GxfComponentTypeId(entity_.context(), type_name, &tid);
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Failed to get type id for type [%s]", type_name);
    return null_handle;
  }
  auto maybe_untyped_handle = entity_.get(tid, name);
  if (!maybe_untyped_handle) {
    GXF_LOG_ERROR("Failed to get handle for [%s] of type [%s] in entity [%s] with error %s",
                  name, type_name, entity_.name(), GxfResultStr(maybe_untyped_handle.error()));
    return null_handle;
  }

  auto maybe_handle = Handle<Component>::Create(maybe_untyped_handle.value());
  if (!maybe_handle) {
    GXF_LOG_ERROR("Failed to convert untyped handle to handle with error %s",
                  GxfResultStr(maybe_handle.error()));
    return null_handle;
  }
  return maybe_handle.value();
}

Expected<Handle<Component>> GraphEntity::try_get(const char* type_name, const char* name) const {
  gxf_tid_t tid;
  const auto result = GxfComponentTypeId(entity_.context(), type_name, &tid);
  if (!isSuccessful(result)) { return Unexpected{result}; }

  auto maybe_untyped_handle = entity_.get(tid, name);
  if (!maybe_untyped_handle) { return ForwardError(maybe_untyped_handle); }

  auto maybe_handle = Handle<Component>::Create(maybe_untyped_handle.value());
  if (!maybe_handle) { return ForwardError(maybe_handle); }

  return maybe_handle.value();
}

Handle<Codelet> GraphEntity::addCodelet(const char* type_name, const char* name,
                                        const std::vector<Arg>& arg_list) {
  if (!codelet_.is_null()) {
    GXF_LOG_ERROR("Graph Entity is already configured with a codelet [%s]", codelet_->name());
    return Handle<Codelet>::Null();
  }
  auto maybe_codelet = createHandle<Codelet>(type_name, name);
  if (!maybe_codelet) {
    GXF_LOG_ERROR("Graph Entity failed to create codelet of type [%s]", type_name);
    return Handle<Codelet>::Null();
  }
  auto codelet = maybe_codelet.value();

  for (auto arg : arg_list) { applyArg(codelet, arg); }

  codelet_ = codelet;
  return codelet;
}

Handle<Component> GraphEntity::addComponent(const char* type_name, const char* name,
                                            const std::vector<Arg>& arg_list) {
  auto maybe_component = createHandle<Component>(type_name, name);
  if (!maybe_component) {
    GXF_LOG_ERROR("Graph Entity failed to create component of type [%s]", type_name);
    return Handle<Component>::Null();
  }
  auto component = maybe_component.value();

  for (auto arg : arg_list) { applyArg(component, arg); }

  components_.emplace(component.get()->cid(), component);
  return component;
}

Handle<Clock> GraphEntity::addClock(const char* type_name, const char* name,
                                    const std::vector<Arg>& arg_list) {
  auto clock = Handle<Clock>::Null();
  if (clocks_.find(name) != clocks_.end()) {
    GXF_LOG_ERROR("Clock with same name [%s] already exists in entity [%s]", name, entity_.name());
    return clock;
  }
  auto maybe_clock = createHandle<Clock>(type_name, name);
  if (!maybe_clock) {
    GXF_LOG_ERROR("Graph Entity failed to create clock of type [%s]", type_name);
    return clock;
  }
  clock = maybe_clock.value();

  for (auto arg : arg_list) { applyArg(clock, arg); }

  clocks_.emplace(name, clock);
  return clock;
}

Handle<Clock> GraphEntity::getClock(const char* name) {
  if (clocks_.empty()) {
    GXF_LOG_ERROR("No clock components found in entity [%s]", entity_.name());
    return Handle<Clock>::Null();
  }

  if (name == nullptr) {
    return clocks_.begin()->second;
  }

  auto it = clocks_.find(name);
  if (it == clocks_.end()) {
    GXF_LOG_ERROR("Clock component with name [%s] not found in entity [%s]", name, entity_.name());
    return Handle<Clock>::Null();
  }

  return it->second;
}

Handle<SchedulingTerm> GraphEntity::addSchedulingTerm(const char* type_name, const char* name,
                                                      const std::vector<Arg>& arg_list) {
  auto term = Handle<SchedulingTerm>::Null();
  if (terms_.find(name) != terms_.end()) {
    GXF_LOG_ERROR("Scheduling term with same name [%s] already exists in entity [%s]", name,
                  entity_.name());
    return term;
  }
  auto maybe_term = createHandle<SchedulingTerm>(type_name, name);
  if (!maybe_term) {
    GXF_LOG_ERROR("Graph Entity failed to create scheduling term of type [%s]", type_name);
    return term;
  }
  term = maybe_term.value();

  for (auto arg : arg_list) { applyArg(term, arg); }

  terms_.emplace(name, term);
  return term;
}

Handle<Transmitter> GraphEntity::addTransmitter(const char* type_name, const char* name,
                                                const std::vector<Arg>& arg_list, bool omit_term) {
  auto tx = Handle<Transmitter>::Null();
  if (tx_queues_.find(name) != tx_queues_.end()) {
    GXF_LOG_ERROR("Transmitter term with same name [%s] already exists in entity [%s]", name,
                  entity_.name());
    return tx;
  }
  auto maybe_tx_name = formatTxName(name);
  if (!maybe_tx_name) { return tx; }
  auto tx_name = maybe_tx_name.value();

  auto maybe_tx = createHandle<Transmitter>(type_name, tx_name.c_str());
  if (!maybe_tx) {
    GXF_LOG_ERROR("Graph Entity failed to create transmitter of type [%s]", type_name);
    return tx;
  }
  tx = maybe_tx.value();

  for (auto arg : arg_list) { applyArg(tx, arg); }

  tx_queues_.emplace(tx_name.c_str(), tx);

  // auto adds a downstream receptive scheduling term
  if (!omit_term) {
    auto term = this->add<DownstreamReceptiveSchedulingTerm>(tx_name.c_str());
    term->setTransmitter(tx);
  }

  std::string full_name = std::string(this->name()) + "/" + tx_name;
  auto result = updatePort(name, full_name);
  if (!result) {
    GXF_LOG_ERROR("Failed to add Transmitter [%s] with error [%s]", tx_name.c_str(),
                  GxfResultStr(result.error()));
  }
  return tx;
}

Handle<Receiver> GraphEntity::addReceiver(const char* type_name, const char* name,
                                          const std::vector<Arg>& arg_list, bool omit_term) {
  auto rx = Handle<Receiver>::Null();
  if (rx_queues_.find(name) != rx_queues_.end()) {
    GXF_LOG_ERROR("Receiver term with same name [%s] already exists in entity [%s]", name,
                  entity_.name());
    return rx;
  }
  auto maybe_rx_name = formatRxName(name);
  if (!maybe_rx_name) { return Handle<Receiver>::Null(); }
  auto rx_name = maybe_rx_name.value();

  auto maybe_rx = createHandle<Receiver>(type_name, rx_name.c_str());
  if (!maybe_rx) {
    GXF_LOG_ERROR("Graph Entity failed to create receiver of type [%s]", type_name);
    return rx;
  }
  rx = maybe_rx.value();

  for (auto arg : arg_list) { applyArg(rx, arg); }

  rx_queues_.emplace(rx_name.c_str(), rx);

  // auto adds a message available scheduling term
  if (!omit_term) {
    auto term = this->add<MessageAvailableSchedulingTerm>(rx_name.c_str());
    term->setReceiver(rx);
  }

  std::string full_name = std::string(this->name()) + "/" + rx_name;
  auto result = updatePort(name, full_name);
  if (!result) {
    GXF_LOG_ERROR("Failed to add Receiver [%s] with error [%s]", rx_name.c_str(),
                  GxfResultStr(result.error()));
  }
  return rx;
}

}  // namespace gxf
}  // namespace nvidia
