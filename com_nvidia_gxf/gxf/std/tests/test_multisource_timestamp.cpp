/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

TEST(MultiSourceTimestamp, verifyAllAvailableSources) {
  MultiSourceTimestamp timestamps;
  std::array<Timestamp, MAX_TIME_DOMAINS> timestamp;
  for (int32_t i = 0; i < MAX_TIME_DOMAINS; i++) {
    timestamp[i].acqtime = rand();
    timestamp[i].pubtime = rand();
    timestamps.push_back(std::make_pair(timestamp[i], static_cast<TimeDomainID>(i)));
  }

  for (int32_t i = 0; i < MAX_TIME_DOMAINS; i++) {
    Expected<Timestamp> maybe_timestamp = getTimestamp(timestamps, static_cast<TimeDomainID>(i));
    if (maybe_timestamp) {
      ASSERT_EQ(timestamp[i].acqtime, maybe_timestamp.value().acqtime);
      ASSERT_EQ(timestamp[i].pubtime, maybe_timestamp.value().pubtime);
    }
  }
}

TEST(MultiSourceTimestamp, verifyPartialSources) {
  MultiSourceTimestamp timestamps;
  std::array<Timestamp, MAX_TIME_DOMAINS> timestamp;
  timestamps.clear();
  for (int32_t i = 0; i < 1; i++) {
    timestamp[i].acqtime = rand();
    timestamp[i].pubtime = rand();
    timestamps.push_back(std::make_pair(timestamp[i], static_cast<TimeDomainID>(i)));
  }

  for (int32_t i = 1; i < MAX_TIME_DOMAINS; i++) {
    Expected<Timestamp> maybe_timestamp = getTimestamp(timestamps, static_cast<TimeDomainID>(i));
    EXPECT_FALSE(maybe_timestamp);
    if (maybe_timestamp) {
      ASSERT_EQ(timestamp[i].acqtime, maybe_timestamp.value().acqtime);
      ASSERT_EQ(timestamp[i].pubtime, maybe_timestamp.value().pubtime);
    }
  }
}

TEST(MultiSourceTimestamp, checkAbsentSources) {
  MultiSourceTimestamp timestamps;
  Timestamp timestamp;

  timestamp.acqtime = rand();
  timestamp.pubtime = rand();
  timestamps.clear();
  timestamps.push_back(std::make_pair(timestamp, TimeDomainID::TSC));

  Expected<Timestamp> maybe_timestamp = getTimestamp(timestamps, TimeDomainID::NTP);
  EXPECT_FALSE(maybe_timestamp);

  maybe_timestamp = getTimestamp(timestamps, TimeDomainID::PTP);
  EXPECT_FALSE(maybe_timestamp);

  maybe_timestamp = getTimestamp(timestamps, TimeDomainID::TSC);
  EXPECT_TRUE(maybe_timestamp);
  if (maybe_timestamp) {
    ASSERT_EQ(timestamp.acqtime, maybe_timestamp.value().acqtime);
    ASSERT_EQ(timestamp.pubtime, maybe_timestamp.value().pubtime);
  }
}

}  // namespace gxf
}  // namespace nvidia
