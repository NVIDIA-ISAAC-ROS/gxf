/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/core/gxf.h"
#include <iostream>
#include "gtest/gtest.h"
#include "common/assert.hpp"

TEST(GxfRegisterComponent,invalid_context)
{
  gxf_tid_t tid{0UL,0UL};
  ASSERT_EQ(GxfRegisterComponent(NULL,tid, "nvidia::gxf::Comp1", ""),GXF_CONTEXT_INVALID);
}

TEST(GxfRegisterComponent,valid_context)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid{0UL,0UL};
  ASSERT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp1", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,re_register_component)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid{0UL,0UL};
  ASSERT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Component", ""),GXF_FACTORY_DUPLICATE_TID);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,valid_gxf_tid)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid{0UL,0UL} ;
  ASSERT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp1", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,duplicate_gxf_tid_name)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid{0UL,0UL} ;
  gxf_tid_t tidl{0UL,1UL};
  ASSERT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfRegisterComponent(context,tidl, "nvidia::gxf::Comp1", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfRegisterComponent(context,tid,"nvidia::gxf::Comp1", ""),GXF_FACTORY_DUPLICATE_TID);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,duplicate_gxf_tid)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid{0UL,0UL} ;
  gxf_tid_t tidl{0UL,0UL};
  ASSERT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfRegisterComponent(context,tidl, "nvidia::gxf::Comp1", ""),GXF_FACTORY_DUPLICATE_TID);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,not_initialised_gxf_tid)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid = GxfTidNull();
  EXPECT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp1", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,name_starting_with_special_Symbols_and_digit)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid = GxfTidNull();
  const char *component_name="_name";
  EXPECT_EQ(GxfRegisterComponent(context,tid,component_name, ""),GXF_SUCCESS);
  tid = {0UL, 1UL};
  const char *component_name_1="1_name";
  EXPECT_EQ(GxfRegisterComponent(context,tid,component_name_1, ""),GXF_SUCCESS);
  tid = {1UL, 1UL};
  const char *component_name_2="@_name";
  EXPECT_EQ(GxfRegisterComponent(context,tid,component_name_2, ""),GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,re_registering_name_with_same_tid)
{
  gxf_context_t context = kNullContext;
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
  gxf_tid_t tid = GxfTidNull();
  EXPECT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp1", ""),GXF_SUCCESS);
  EXPECT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::Comp1", ""),GXF_FACTORY_DUPLICATE_TID);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,null_value_for_base_class)
{
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context),GXF_SUCCESS);
  gxf_tid_t tid = GxfTidNull();
  ASSERT_EQ(GxfRegisterComponent(context,tid, "nvidia::gxf::NewComponent", ""),GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,invalid_class_name)
{
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context),GXF_SUCCESS);
  gxf_tid_t tid = GxfTidNull();
  ASSERT_EQ(GxfRegisterComponent(context,tid,"nvidia::gxf::NewComp1","invalid_class"),GXF_FACTORY_UNKNOWN_CLASS_NAME);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}

TEST(GxfRegisterComponent,valid_class)
{
  gxf_context_t context = kNullContext;
  ASSERT_EQ(GxfContextCreate(&context),GXF_SUCCESS);
  gxf_tid_t tid = GxfTidNull();
  ASSERT_EQ(GxfRegisterComponent(context,tid,"nvidia::gxf::NewComponent",""),GXF_SUCCESS);
  ASSERT_EQ(GxfContextDestroy(context),GXF_SUCCESS);
}