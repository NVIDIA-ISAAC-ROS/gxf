/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/synchronization.hpp"
#include "gxf/std/timestamp.hpp"

#include "gtest/gtest.h"

namespace nvidia {
namespace gxf {

class TestSynchronization : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(GxfContextCreate(&context_), GXF_SUCCESS);
    ASSERT_EQ(GxfSetSeverity(&context_, GXF_SEVERITY_VERBOSE), GXF_SUCCESS);

    constexpr const char* kExtensions[] = {
      "gxf/std/libgxf_std.so",
    };
    const GxfLoadExtensionsInfo info{kExtensions, 1, nullptr, 0, nullptr};

    ASSERT_EQ(GxfLoadExtensions(context_, &info), GXF_SUCCESS);

    gxf_uid_t eid;
    const GxfEntityCreateInfo entity_create_info = {0};
    ASSERT_EQ(GxfCreateEntity(context_, &entity_create_info, &eid), GXF_SUCCESS);

    gxf_tid_t sync_tid;
    void* sync_pointer;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::Synchronization", &sync_tid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentAdd(context_, eid, sync_tid, "sync", &sync_cid_), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentPointer(context_, sync_cid_, sync_tid, &sync_pointer), GXF_SUCCESS);
    sync_ = static_cast<Synchronization*>(sync_pointer);

    gxf_tid_t rx_tid;
    gxf_uid_t rx1_cid;
    void* rx1_pointer;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::DoubleBufferReceiver", &rx_tid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentAdd(context_, eid, rx_tid, "rx1", &rx1_cid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentPointer(context_, rx1_cid, rx_tid, &rx1_pointer), GXF_SUCCESS);
    rx1_ = static_cast<DoubleBufferReceiver*>(rx1_pointer);

    gxf_uid_t rx2_cid;
    void* rx2_pointer;
    ASSERT_EQ(GxfComponentAdd(context_, eid, rx_tid, "rx2", &rx2_cid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentPointer(context_, rx2_cid, rx_tid, &rx2_pointer), GXF_SUCCESS);
    rx2_ = static_cast<DoubleBufferReceiver*>(rx2_pointer);


    gxf_tid_t tx_tid;
    gxf_uid_t tx1_cid;
    void* tx1_pointer;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::DoubleBufferTransmitter", &tx_tid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentAdd(context_, eid, tx_tid, "tx1", &tx1_cid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentPointer(context_, tx1_cid, tx_tid, &tx1_pointer), GXF_SUCCESS);
    tx1_ = static_cast<DoubleBufferTransmitter*>(tx1_pointer);

    gxf_uid_t tx2_cid;
    void* tx2_pointer;
    ASSERT_EQ(GxfComponentAdd(context_, eid, tx_tid, "tx2", &tx2_cid), GXF_SUCCESS);
    ASSERT_EQ(GxfComponentPointer(context_, tx2_cid, tx_tid, &tx2_pointer), GXF_SUCCESS);
    tx2_ = static_cast<DoubleBufferTransmitter*>(tx2_pointer);

    YAML::Node rx_handles;
    rx_handles.push_back("rx1");
    rx_handles.push_back("rx2");
    ASSERT_EQ(GxfParameterSetFromYamlNode(context_, sync_cid_, "inputs", &rx_handles, ""), GXF_SUCCESS);

    YAML::Node tx_handles;
    tx_handles.push_back("tx1");
    tx_handles.push_back("tx2");
    ASSERT_EQ(GxfParameterSetFromYamlNode(context_, sync_cid_, "outputs", &tx_handles, ""), GXF_SUCCESS);

    ASSERT_EQ(sync_->initialize(), GXF_SUCCESS);
    ASSERT_EQ(rx1_->initialize(), GXF_SUCCESS);
    ASSERT_EQ(rx2_->initialize(), GXF_SUCCESS);
    ASSERT_EQ(tx1_->initialize(), GXF_SUCCESS);
    ASSERT_EQ(tx2_->initialize(), GXF_SUCCESS);
  }

  void TearDown() override {
    ASSERT_EQ(sync_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(rx1_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(rx2_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(tx1_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(tx2_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(GxfContextDestroy(context_), GXF_SUCCESS);
  }

  gxf_uid_t sync_cid_;
  gxf_context_t context_;
  Synchronization* sync_;
  DoubleBufferReceiver* rx1_;
  DoubleBufferReceiver* rx2_;
  DoubleBufferTransmitter* tx1_;
  DoubleBufferTransmitter* tx2_;
};

TEST_F(TestSynchronization, canCreate) {
  // do nothing, just make sure we can instantiate everything
}

// Factory method to create message entity
Expected<Entity> createMessage(gxf_context_t context, int64_t timestamp) {
  Expected<Entity> message = Entity::New(context);
  if (!message) { return message; }

  auto msg = message.value().add<Timestamp>("timestamp");
  if (!msg) { return ForwardError(msg); }
  msg.value()->acqtime = timestamp;

  return message;
}

TEST_F(TestSynchronization, simple) {
  // make sure we can sync messages with same timestamp,
  // without any threshold
  auto message1 = createMessage(context_, 0);
  auto message2 = createMessage(context_, 0);
  ASSERT_TRUE(message1);
  ASSERT_TRUE(message2);

  ASSERT_EQ(rx1_->push_abi(message1.value().eid()), GXF_SUCCESS);
  ASSERT_EQ(rx2_->push_abi(message2.value().eid()), GXF_SUCCESS);
  ASSERT_TRUE(rx1_->sync());
  ASSERT_TRUE(rx2_->sync());

  ASSERT_EQ(sync_->tick(), GXF_SUCCESS);

  ASSERT_TRUE(tx1_->sync());
  ASSERT_TRUE(tx2_->sync());
  ASSERT_EQ(tx1_->size(), 1);
  ASSERT_EQ(tx2_->size(), 1);
  ASSERT_TRUE(tx1_->pop());
  ASSERT_TRUE(tx2_->pop());
}

TEST_F(TestSynchronization, WithinThreshold) {
  // msgs are within threshold, sync point is present
  // msgs are fwded to tx queues
  auto message1 = createMessage(context_, 0);
  auto message2 = createMessage(context_, 5);
  ASSERT_TRUE(message1);
  ASSERT_TRUE(message2);

  ASSERT_EQ(rx1_->push_abi(message1.value().eid()), GXF_SUCCESS);
  ASSERT_EQ(rx2_->push_abi(message2.value().eid()), GXF_SUCCESS);
  ASSERT_TRUE(rx1_->sync());
  ASSERT_TRUE(rx2_->sync());

  ASSERT_EQ(GxfParameterSetInt64(context_, sync_cid_, "sync_threshold", 5), GXF_SUCCESS);
  ASSERT_EQ(sync_->tick(), GXF_SUCCESS);

  ASSERT_TRUE(tx1_->sync());
  ASSERT_TRUE(tx2_->sync());
  ASSERT_EQ(tx1_->size(), 1);
  ASSERT_EQ(tx2_->size(), 1);
  ASSERT_TRUE(tx1_->pop());
  ASSERT_TRUE(tx2_->pop());
}

TEST_F(TestSynchronization, OutsideThreshold) {
  // msgs are outside threshold, no sync point present
  // msgs are considered stale and dropped, tx queues are empty
  auto message1 = createMessage(context_, 0);
  auto message2 = createMessage(context_, 5);
  ASSERT_TRUE(message1);
  ASSERT_TRUE(message2);

  ASSERT_EQ(rx1_->push_abi(message1.value().eid()), GXF_SUCCESS);
  ASSERT_EQ(rx2_->push_abi(message2.value().eid()), GXF_SUCCESS);
  ASSERT_TRUE(rx1_->sync());
  ASSERT_TRUE(rx2_->sync());

  ASSERT_EQ(GxfParameterSetInt64(context_, sync_cid_, "sync_threshold", 3), GXF_SUCCESS);
  ASSERT_EQ(sync_->tick(), GXF_SUCCESS);

  ASSERT_TRUE(tx1_->sync());
  ASSERT_TRUE(tx2_->sync());
  ASSERT_EQ(tx1_->size(), 0);
  ASSERT_EQ(tx2_->size(), 0);
  ASSERT_EQ(rx1_->size(), 0);
  ASSERT_EQ(rx2_->size(), 1); // rx2 is still within threshold, preserved
  ASSERT_TRUE(rx2_->pop());
}

TEST_F(TestSynchronization, BackstageOnly) {
  // msgs only in backstage, main stage empty
  // rx queues are not synced
  auto message1 = createMessage(context_, 0);
  auto message2 = createMessage(context_, 0);
  ASSERT_TRUE(message1);
  ASSERT_TRUE(message2);

  ASSERT_EQ(rx1_->push_abi(message1.value().eid()), GXF_SUCCESS);
  ASSERT_EQ(rx2_->push_abi(message2.value().eid()), GXF_SUCCESS);

  ASSERT_EQ(sync_->tick(), GXF_CONTRACT_MESSAGE_NOT_AVAILABLE);

  ASSERT_TRUE(tx1_->sync());
  ASSERT_TRUE(tx2_->sync());
  ASSERT_EQ(tx1_->size(), 0);
  ASSERT_EQ(tx2_->size(), 0);
  ASSERT_EQ(rx1_->size(), 0);
  ASSERT_EQ(rx2_->size(), 0);
}

TEST_F(TestSynchronization, BackstageAndMainstage) {
  // msgs are in backstage and mainstage,
  // only messages in mainstage are synced and forwarded
  auto message1 = createMessage(context_, 0);
  auto message2 = createMessage(context_, 0);
  ASSERT_TRUE(message1);
  ASSERT_TRUE(message2);

  ASSERT_EQ(rx1_->push_abi(message1.value().eid()), GXF_SUCCESS);
  ASSERT_EQ(rx2_->push_abi(message2.value().eid()), GXF_SUCCESS);
  ASSERT_TRUE(rx1_->sync());
  ASSERT_TRUE(rx2_->sync());
  auto message3 = createMessage(context_, 0);
  auto message4 = createMessage(context_, 0);
  ASSERT_TRUE(message3);
  ASSERT_TRUE(message4);
  ASSERT_EQ(rx1_->push_abi(message3.value().eid()), GXF_SUCCESS);
  ASSERT_EQ(rx2_->push_abi(message4.value().eid()), GXF_SUCCESS);

  ASSERT_EQ(sync_->tick(), GXF_SUCCESS);

  ASSERT_TRUE(tx1_->sync());
  ASSERT_TRUE(tx2_->sync());
  ASSERT_EQ(tx1_->size(), 1);
  ASSERT_EQ(tx2_->size(), 1);
  ASSERT_EQ(rx1_->size(), 0);
  ASSERT_EQ(rx2_->size(), 0);
  ASSERT_EQ(rx1_->back_size(), 1);
  ASSERT_EQ(rx2_->back_size(), 1);
  ASSERT_TRUE(tx1_->pop());
  ASSERT_TRUE(tx2_->pop());
}


}  // namespace gxf
}  // namespace nvidia
