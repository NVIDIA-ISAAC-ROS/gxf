/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <chrono>
#include <cstring>
#include <thread>

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/vault.hpp"

namespace {

constexpr const char* kGxeManifestFilename = "gxf/gxe/manifest.yaml";

}  // namespace

class CallbackTester {
 public:
  CallbackTester() {
    notify_callback_ = std::bind(&CallbackTester::notifyCallback, this);
  }

  nvidia::gxf::Vault::CallbackType& getNotifyCallback() {
    return notify_callback_;
  }
  const int getNotifyCount() const { return notify_count_; }

 private:
  int notify_count_ = 0;
  nvidia::gxf::Vault::CallbackType notify_callback_;

 private:
  void notifyCallback() {
    GXF_LOG_INFO("Notify callback calling [%d] times", ++notify_count_);
  }
};

class Vault_Test : public ::testing::Test {
 public:
  void SetUp() {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context_));
    GXF_ASSERT_SUCCESS(GxfSetSeverity(context_, static_cast<gxf_severity_t>(4)));
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context_, &info_));
    GXF_ASSERT_SUCCESS(GxfGraphLoadFile(context_, "gxf/test/apps/test_vault.yaml"));

    GXF_ASSERT_SUCCESS(GxfEntityFind(context_, "vault_entity", &eid_));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::Vault", &tid_));
    GXF_ASSERT_SUCCESS(GxfComponentFind(context_, eid_, tid_, "vault", nullptr, &cid_));
    callback_tester_ = new CallbackTester();

    gxf_uid_t eid_tx;
    gxf_uid_t cid_count_st;
    gxf_tid_t tid_count_st;
    GXF_ASSERT_SUCCESS(GxfEntityFind(context_, "tx", &eid_tx));
    GXF_ASSERT_SUCCESS(GxfComponentTypeId(context_, "nvidia::gxf::CountSchedulingTerm", &tid_count_st));
    GXF_ASSERT_SUCCESS(GxfComponentFind(context_, eid_tx, tid_count_st, nullptr, nullptr, &cid_count_st));
    GXF_ASSERT_SUCCESS(GxfParameterGetInt64(context_, cid_count_st, "count", &tx_steps_));
  }

  void TearDown() {
    GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context_));
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context_));
    delete callback_tester_;
  }

 protected:
  gxf_context_t context_ = kNullContext;
  const GxfLoadExtensionsInfo info_{nullptr, 0, &kGxeManifestFilename, 1, nullptr};
  gxf_uid_t eid_ = kNullUid;
  gxf_uid_t cid_ = kNullUid;
  gxf_tid_t tid_;
  CallbackTester* callback_tester_ = nullptr;
  int64_t tx_steps_ = 0;
};

TEST_F(Vault_Test, VaultSetCallback) {
  auto maybe_vault = nvidia::gxf::Handle<nvidia::gxf::Vault>::Create(context_, cid_);
  ASSERT_TRUE(maybe_vault.has_value());
  auto vault = maybe_vault.value();
  vault->setCallback(callback_tester_->getNotifyCallback());

  GXF_ASSERT_SUCCESS(GxfGraphActivate(context_));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context_));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context_));

  GXF_ASSERT_EQ(callback_tester_->getNotifyCount(), tx_steps_);
}

TEST_F(Vault_Test, VaultForgetToSetCallback) {
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context_));
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context_));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context_));

  GXF_ASSERT_EQ(callback_tester_->getNotifyCount(), 0);
}