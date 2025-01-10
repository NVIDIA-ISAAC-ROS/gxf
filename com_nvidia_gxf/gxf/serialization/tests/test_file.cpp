/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gtest/gtest.h"

#include <cstring>
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/serialization/file.hpp"
#include "gxf/serialization/file_stream.hpp"
#include "gxf/std/allocator.hpp"

namespace nvidia {
namespace gxf {
namespace test {

namespace {

constexpr size_t kBufferSizeSmall = 1000;
constexpr size_t kBufferSizeMedium = 10000;
constexpr size_t kBufferSizeLarge = 100000;

class TestFile : public ::testing::Test {
 protected:
  File* file;

 private:
  void SetUp() override {
    ASSERT_EQ(GxfContextCreate(&context_), GXF_SUCCESS);

    const char* manifests[] = {
      "gxf/gxe/manifest.yaml",
    };
    const GxfLoadExtensionsInfo extension_info = { nullptr, 0, manifests, 1, nullptr };
    ASSERT_EQ(GxfLoadExtensions(context_, &extension_info), GXF_SUCCESS);

    const GxfEntityCreateInfo entity_info = { "entity", 0 };
    ASSERT_EQ(GxfCreateEntity(context_, &entity_info, &eid_), GXF_SUCCESS);

    gxf_tid_t tid1;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::UnboundedAllocator", &tid1), GXF_SUCCESS);
    gxf_uid_t cid1;
    ASSERT_EQ(GxfComponentAdd(context_, eid_, tid1, "allocator", &cid1), GXF_SUCCESS);
    void* component1;
    ASSERT_EQ(GxfComponentPointer(context_, cid1, tid1, &component1), GXF_SUCCESS);
    allocator_ = static_cast<Allocator*>(component1);
    ASSERT_EQ(allocator_->initialize(), GXF_SUCCESS);

    gxf_tid_t tid2;
    ASSERT_EQ(GxfComponentTypeId(context_, "nvidia::gxf::File", &tid2), GXF_SUCCESS);
    gxf_uid_t cid2;
    ASSERT_EQ(GxfComponentAdd(context_, eid_, tid2, "file", &cid2), GXF_SUCCESS);
    void* component2;
    ASSERT_EQ(GxfComponentPointer(context_, cid2, tid2, &component2), GXF_SUCCESS);
    file = static_cast<File*>(component2);
    ASSERT_EQ(GxfParameterSetHandle(context_, cid2, "allocator", cid1), GXF_SUCCESS);
    ASSERT_EQ(file->initialize(), GXF_SUCCESS);
  }

  void TearDown() override {
    ASSERT_EQ(file->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(allocator_->deinitialize(), GXF_SUCCESS);
    ASSERT_EQ(GxfEntityDestroy(context_, eid_), GXF_SUCCESS);
    ASSERT_EQ(GxfGraphDeactivate(context_), GXF_SUCCESS);
    ASSERT_EQ(GxfContextDestroy(context_), GXF_SUCCESS);
  }

  gxf_context_t context_;
  gxf_uid_t eid_;
  Allocator* allocator_;
};

}  // namespace

TEST_F(TestFile, Open) {
  const std::string file_path = "/tmp/test_file_open";
  ASSERT_FALSE(file->isOpen());
  ASSERT_TRUE(file->open(file_path.c_str()));
  ASSERT_TRUE(file->isOpen());
}

TEST_F(TestFile, Close) {
  const std::string file_path = "/tmp/test_file_close";
  ASSERT_FALSE(file->isOpen());
  ASSERT_TRUE(file->open(file_path.c_str()));
  ASSERT_TRUE(file->isOpen());
  ASSERT_TRUE(file->close());
  ASSERT_FALSE(file->isOpen());
}

TEST_F(TestFile, Write) {
  const std::string file_path = "/tmp/test_file_write";
  byte buffer[kBufferSizeMedium];
  std::memset(buffer, 0xAA, sizeof(buffer));
  ASSERT_TRUE(file->open(file_path.c_str()));
  auto result = file->write(buffer, sizeof(buffer));
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), sizeof(buffer));
}

TEST_F(TestFile, Read) {
  const std::string file_path = "/tmp/test_file_read";
  byte buffer1[kBufferSizeSmall];
  std::memset(buffer1, 0xAA, sizeof(buffer1));
  ASSERT_TRUE(file->open(file_path.c_str()));
  ASSERT_TRUE(file->write(buffer1, sizeof(buffer1)));

  ASSERT_TRUE(file->flush());
  ASSERT_TRUE(file->seek(0));

  byte buffer2[kBufferSizeSmall];
  std::memset(buffer2, 0x00, sizeof(buffer2));
  ASSERT_TRUE(file->isReadAvailable());
  auto result = file->read(buffer2, sizeof(buffer2));
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), sizeof(buffer2));
  ASSERT_EQ(std::memcmp(buffer2, buffer1, sizeof(buffer1)), 0);
}

TEST_F(TestFile, Path) {
  const std::string file_path = "/tmp/test_file_path";
  ASSERT_EQ(file->path(), nullptr);
  ASSERT_TRUE(file->open(file_path.c_str()));
  ASSERT_EQ(file->path(), file_path);
  ASSERT_TRUE(file->close());
  ASSERT_EQ(file->path(), file_path);
}

TEST_F(TestFile, Mode) {
  const std::string file_path = "/tmp/test_file_mode";
  const std::string mode_write = "w";
  const std::string mode_read = "r";
  ASSERT_TRUE(file->open(file_path.c_str(), mode_write.c_str()));
  ASSERT_EQ(file->mode(), mode_write);
  ASSERT_TRUE(file->close());
  ASSERT_TRUE(file->open(file_path.c_str(), mode_read.c_str()));
  ASSERT_EQ(file->mode(), mode_read);
}

TEST_F(TestFile, Rename) {
  const std::string file_path_old = "/tmp/test_file_rename_old";
  const std::string file_path_new = "/tmp/test_file_rename_new";
  byte buffer1[kBufferSizeSmall];
  std::memset(buffer1, 0xAA, sizeof(buffer1));
  ASSERT_TRUE(file->open(file_path_old.c_str()));
  ASSERT_EQ(file->path(), file_path_old);
  ASSERT_TRUE(file->isWriteAvailable());
  ASSERT_TRUE(file->write(buffer1, sizeof(buffer1)));
  ASSERT_TRUE(file->flush());
  ASSERT_TRUE(file->seek(0));

  ASSERT_TRUE(file->rename(file_path_new.c_str()));
  ASSERT_EQ(file->path(), file_path_new);

  byte buffer2[kBufferSizeSmall];
  std::memset(buffer2, 0x00, sizeof(buffer2));
  ASSERT_TRUE(file->read(buffer2, sizeof(buffer2)));
  ASSERT_EQ(std::memcmp(buffer2, buffer1, sizeof(buffer1)), 0);
}

TEST_F(TestFile, Timestamp) {
  const std::string file_path_old = "/tmp/test_file_timestamp";
  const std::string file_path_new = "/tmp/1970-01-01_00-00-00_test_file_timestamp";
  ASSERT_TRUE(file->open(file_path_old.c_str()));
  ASSERT_TRUE(file->close());
  ASSERT_TRUE(file->addTimestamp(0, true));
  ASSERT_EQ(file->path(), file_path_new);
}

TEST_F(TestFile, InvalidFileOpen) {
  ASSERT_FALSE(file->open(nullptr));
  ASSERT_FALSE(file->open(""));
  ASSERT_FALSE(file->open("~/DummyFile"));
}

TEST_F(TestFile, InvalidFileClose) {
  ASSERT_FALSE(file->close());
}

TEST_F(TestFile, InvalidFileFlush) {
  ASSERT_FALSE(file->flush());
}

TEST_F(TestFile, FilePosition) {
  const std::string file_path_old = "/tmp/test_file_timestamp";
  const std::string file_path_new = "/tmp/1970-01-01_00-00-00_test_file_timestamp";
  ASSERT_TRUE(file->open(file_path_old.c_str()));
  size_t fileSize = 0;
  ASSERT_TRUE(file->tell() == fileSize);
}

TEST(TestFileStrean, FileStream) {
  std::string path = "DummyFile";
  auto file_stream_ = FileStream("", path + nvidia::gxf::FileStream::kIndexFileExtension);
  ASSERT_TRUE(file_stream_.open());
  ASSERT_TRUE(file_stream_.setWriteOffset(0));
  byte buffer[kBufferSizeSmall];
  std::memset(buffer, 0xAA, sizeof(buffer));
  ASSERT_TRUE(file_stream_.write(buffer, sizeof(buffer)));
  auto offset = file_stream_.getWriteOffset();
  ASSERT_TRUE(file_stream_.flush());
  ASSERT_TRUE(offset == kBufferSizeSmall);
  file_stream_.clear();
  ASSERT_TRUE(file_stream_.close());
}

TEST(TestFileStrean, FileStreamRead) {
  std::string path = "DummyFile";
  auto file_write_stream_ = FileStream("", path + nvidia::gxf::FileStream::kIndexFileExtension);
  ASSERT_TRUE(file_write_stream_.open());
  byte buffer[kBufferSizeSmall];
  std::memset(buffer, 0xAA, sizeof(buffer));
  ASSERT_TRUE(file_write_stream_.write(buffer, sizeof(buffer)));

  auto file_read_stream_ = FileStream(path + nvidia::gxf::FileStream::kIndexFileExtension, "");
  ASSERT_TRUE(file_read_stream_.open());
  constexpr auto fileOffsetToSet = 100;
  ASSERT_TRUE(file_read_stream_.setReadOffset(fileOffsetToSet));
  const size_t offset = file_read_stream_.getReadOffset().value();
  ASSERT_TRUE(offset == fileOffsetToSet);
  ASSERT_TRUE(file_read_stream_.close());
}

}  // namespace test
}  // namespace gxf
}  // namespace nvidia
