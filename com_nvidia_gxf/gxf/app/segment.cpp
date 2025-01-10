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

#include "gxf/app/segment.hpp"

#include <string>
#include <vector>

#include "gxf/std/connection.hpp"
#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"
#include "gxf/std/event_based_scheduler.hpp"
#include "gxf/std/greedy_scheduler.hpp"
#include "gxf/std/multi_thread_scheduler.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/ucx/ucx_component_serializer.hpp"
#include "gxf/ucx/ucx_context.hpp"
#include "gxf/ucx/ucx_receiver.hpp"
#include "gxf/ucx/ucx_serialization_buffer.hpp"
#include "gxf/ucx/ucx_transmitter.hpp"

namespace nvidia {
namespace gxf {

static const constexpr gxf_tid_t TransmitterTID{0xc30cc60f0db2409d, 0x92b6b2db92e02cce};
static const constexpr gxf_tid_t ReceiverTID{0xa47d2f62245f40fc, 0x90b75dc78ff2437e};

Expected<void> Segment::connect(GraphEntityPtr& source, GraphEntityPtr& target,
                                PortPair port_pair) {
  if (!connect_entity_) { connect_entity_ = createGraphEntity("ConnectionEntity_" + name_); }

  auto tx_key = port_pair.tx;
  auto rx_key = port_pair.rx;

  GXF_LOG_DEBUG("Attempting to connect Tx [%s] from [%s] with Rx [%s] from [%s]", tx_key.c_str(),
                source->name(), rx_key.c_str(), target->name());

  // Check if Tx and Rx components have already been added. If yes, this connection would be 1:m or
  // m:1
  auto tx_name = UNWRAP_OR_RETURN(source->formatTxName(tx_key.c_str()));
  auto rx_name = UNWRAP_OR_RETURN(target->formatRxName(rx_key.c_str()));

  Handle<Transmitter> tx;
  auto maybe_tx = source->try_get<DoubleBufferTransmitter>(tx_name.c_str());
  if (!maybe_tx) {
    tx = source->addTransmitter<DoubleBufferTransmitter>(tx_key.c_str());
  } else {
    tx = maybe_tx.value();
  }

  Handle<Receiver> rx;
  auto maybe_rx = target->try_get<DoubleBufferReceiver>(rx_name.c_str());
  if (!maybe_rx) {
    rx = target->addReceiver<DoubleBufferReceiver>(rx_key.c_str());
  } else {
    rx = maybe_rx.value();
  }

  std::string cx_name("Connection_" + std::string(source->name()) + "_" +
                      std::string(target->name()));
  Handle<Connection> cx = connect_entity_->add<Connection>(cx_name.c_str());
  GXF_RETURN_IF_ERROR(cx->setReceiver(rx));
  GXF_RETURN_IF_ERROR(cx->setTransmitter(tx));

  return Success;
}

Expected<void> Segment::connect(GraphEntityPtr& source, GraphEntityPtr& target) {
  if (!connect_entity_) { connect_entity_ = createGraphEntity("ConnectionEntity_" + name_); }

  auto source_codelet = source->get_codelet();
  auto tx_info_list = UNWRAP_OR_RETURN(source_codelet->getParametersOfType<Handle<Transmitter>>());

  if (tx_info_list.size() == 0) {
    GXF_LOG_ERROR("Source codelet [%s] does not have parameter of Transmitter type.",
                  source->name());
    return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
  }

  if (tx_info_list.size() > 1) {
    GXF_LOG_ERROR("More than one transmitter parameter found in [%s]. Please provide port mapping",
                  source->name());
    return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
  }

  auto target_codelet = target->get_codelet();
  auto rx_info_list = UNWRAP_OR_RETURN(target_codelet->getParametersOfType<Handle<Receiver>>());

  if (rx_info_list.size() == 0) {
    GXF_LOG_ERROR("Target codelet [%s] does not have parameter of Receiver type", target->name());
    return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
  }

  if (rx_info_list.size() > 1) {
    GXF_LOG_ERROR("More than one receiver parameter found in [%s]. Please provide port mapping",
                  target->name());
    return Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
  }

  // Check if Tx and Rx components have already been added. If yes, this connection would be 1:m or
  // m:1
  Handle<Transmitter> tx;
  auto maybe_tx = source->try_get<DoubleBufferTransmitter>(tx_info_list[0].key.c_str());
  if (!maybe_tx) {
    tx = source->addTransmitter<DoubleBufferTransmitter>(tx_info_list[0].key.c_str());
  } else {
    tx = maybe_tx.value();
  }

  Handle<Receiver> rx;
  auto maybe_rx = target->try_get<DoubleBufferReceiver>(rx_info_list[0].key.c_str());
  if (!maybe_rx) {
    rx = target->addReceiver<DoubleBufferReceiver>(rx_info_list[0].key.c_str());
  } else {
    rx = maybe_rx.value();
  }

  std::string cx_name("Connection_" + std::string(source->name()) + "_" +
                      std::string(target->name()));
  GXF_LOG_DEBUG("Connecting Tx [%s] from [%s] with Rx [%s] from [%s]", tx->name(), source->name(),
                rx->name(), target->name());
  Handle<Connection> cx = connect_entity_->add<Connection>(cx_name.c_str());
  GXF_RETURN_IF_ERROR(cx->setReceiver(rx));
  GXF_RETURN_IF_ERROR(cx->setTransmitter(tx));

  return Success;
}

// Adds a connection between two entities with many : many tx and rx
Expected<void> Segment::connect(GraphEntityPtr& source, GraphEntityPtr& target,
                                std::vector<PortPair> port_pairs) {
  for (const auto& pair : port_pairs) { RETURN_IF_ERROR(connect(source, target, pair)); }

  return Success;
}

Expected<void> Segment::createNetworkContext() {
  network_ctx_ = createGraphEntity("NetworkContext");
  auto allocator = network_ctx_->add<UnboundedAllocator>("Allocator");
  auto component_serializer = network_ctx_->add<UcxComponentSerializer>("ComponentSerializer");
  auto entity_serializer = network_ctx_->add<UcxEntitySerializer>("EntitySerializer");
  auto ucx_context = network_ctx_->add<UcxContext>("UcxContext");
  RETURN_IF_ERROR(component_serializer->setParameter("allocator", allocator));
  RETURN_IF_ERROR(entity_serializer->add_serializer(component_serializer));
  return ucx_context->setParameter("serializer", entity_serializer);
}

Handle<Component> Segment::createFromProxy(ProxyComponent& component, GraphEntityPtr& entity) {
  gxf_tid_t tid;
  gxf_uid_t cid;
  auto handle = Handle<Component>::Null();

  gxf_result_t result = GxfComponentTypeId(context_, component.type_name().c_str(), &tid);
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Typename [%s] not found. Is this type registered?",
                  component.type_name().c_str());
    return handle;
  }

  result = GxfComponentAdd(context_, entity->eid(), tid, component.name().c_str(), &cid);
  if (!isSuccessful(result)) {
    GXF_LOG_ERROR("Failed to add component of type [%s] with error %s",
                  component.type_name().c_str(), GxfResultStr(result));
    return handle;
  }

  auto maybe_handle = Handle<Component>::Create(context_, cid);
  if (!maybe_handle) { return handle; }
  handle = maybe_handle.value();

  for (auto& arg : component.args()) { applyArg(handle, arg); }

  return handle;
}

Handle<Scheduler> Segment::setScheduler(const SchedulerType& scheduler, std::vector<Arg> arg_list) {
  if (!scheduler_entity_) {
    scheduler_entity_ = createGraphEntity("SchedulerEntity_" + name_);
  } else {
    GXF_LOG_ERROR("Scheduler has already been configured!");
    return {};
  }

  // Check if arg_list has a clock parameter
  auto it = std::find_if(arg_list.begin(), arg_list.end(), [](Arg a) {
                        return a.key() == std::string("clock"); });

  if (it == arg_list.end()) {
      Handle<Clock> clock;
      if (clock_entity_) {
        clock = clock_entity_->getClock();
      } else {
        clock_entity_ = createGraphEntity("ClockEntity_" + name_);
        clock = clock_entity_->add<RealtimeClock>("clock");
      }
      auto clk = Arg("clock", clock);
      arg_list.push_back(clk);
  }

  Handle<Scheduler> handle;
  switch (scheduler) {
    case SchedulerType::kGreedy: {
      handle = scheduler_entity_->add<GreedyScheduler>("Greedy", arg_list);
    } break;
    case SchedulerType::kMultiThread: {
      handle = scheduler_entity_->add<MultiThreadScheduler>("MultiThread", arg_list);
    } break;
    case SchedulerType::KEventBased: {
      handle = scheduler_entity_->add<EventBasedScheduler>("EventBased", arg_list);
    } break;
    default: {
      GXF_LOG_ERROR("Unsupported SchedulerType selected");
    } break;
  }

  return handle;
}

}  // namespace gxf
}  // namespace nvidia
