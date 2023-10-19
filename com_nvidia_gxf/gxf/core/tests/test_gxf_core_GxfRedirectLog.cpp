/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string.h>
#include <iostream>
#include "common/assert.hpp"
#include "gxf/core/gxf.h"

#include "gtest/gtest.h"

namespace {
constexpr const char* kStdExtensionFilename = "gxf/std/libgxf_std.so";
constexpr const char* testString = "This is a test string.";
constexpr const char* filepath = "/tmp/test_log";
}  // namespace

class GxfRedirectLogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    pFile = fopen(filepath, "wr");
  }

  void TearDown() override { GXF_ASSERT_SUCCESS(GxfContextDestroy(context)); }

  std::string GetFileContent() {
    std::string fileContent;
    std::ifstream file;
    file.open(filepath);
    getline(file, fileContent);
    file.close();
    return fileContent;
  }

  gxf_context_t context = kNullContext;
  std::FILE* pFile;
};

TEST_F(GxfRedirectLogTest, InvalidContext) {
  GXF_ASSERT_EQ(GxfRedirectLog(nullptr, pFile), GXF_CONTEXT_INVALID);
}

TEST_F(GxfRedirectLogTest, Error) {
  GXF_ASSERT_SUCCESS(GxfRedirectLog(context, pFile));
  GXF_LOG_ERROR("%s", testString);
  std::string fileContent = GetFileContent();
  GXF_ASSERT_NE(fileContent.find(std::string(testString)), std::string::npos);
  GXF_ASSERT_NE(fileContent.find(std::string("ERROR")), std::string::npos);
}

TEST_F(GxfRedirectLogTest, Debug) {
  // Default severity is INFO so debug level wont be logged.
  GXF_ASSERT_SUCCESS(GxfSetSeverity(context, GXF_SEVERITY_DEBUG));
  GXF_ASSERT_SUCCESS(GxfRedirectLog(context, pFile));
  GXF_LOG_DEBUG("%s", testString);
  std::string fileContent = GetFileContent();
  GXF_ASSERT_NE(fileContent.find(std::string(testString)), std::string::npos);
  GXF_ASSERT_NE(fileContent.find(std::string("DEBUG")), std::string::npos);
}

TEST_F(GxfRedirectLogTest, Info) {
  GXF_ASSERT_SUCCESS(GxfRedirectLog(context, pFile));
  GXF_LOG_INFO("%s", testString);
  std::string fileContent = GetFileContent();
  GXF_ASSERT_NE(fileContent.find(std::string(testString)), std::string::npos);
  GXF_ASSERT_NE(fileContent.find(std::string("INFO")), std::string::npos);
}

TEST_F(GxfRedirectLogTest, Warning) {
  GXF_ASSERT_SUCCESS(GxfRedirectLog(context, pFile));
  GXF_LOG_WARNING("%s", testString);
  std::string fileContent = GetFileContent();
  GXF_ASSERT_NE(fileContent.find(std::string(testString)), std::string::npos);
  GXF_ASSERT_NE(fileContent.find(std::string("WARN")), std::string::npos);
}

TEST_F(GxfRedirectLogTest, Panic) {
  GXF_ASSERT_SUCCESS(GxfRedirectLog(context, pFile));
  EXPECT_DEATH(GXF_PANIC(testString), ".*");
  std::string fileContent = GetFileContent();
  GXF_ASSERT_NE(fileContent.find(std::string(testString)), std::string::npos);
  GXF_ASSERT_NE(fileContent.find(std::string("PANIC")), std::string::npos);
}

TEST_F(GxfRedirectLogTest, NullPointer) {
  GXF_ASSERT_EQ(GxfRedirectLog(context, nullptr), GXF_SUCCESS);
  GXF_LOG_INFO("%s", testString);
}