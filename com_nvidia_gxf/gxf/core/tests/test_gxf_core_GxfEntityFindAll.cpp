/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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

TEST(GxfEntityFindAll,null_context)
{
    gxf_context_t context = kNullContext;
    const char* Entity_Name_1="Entity_1";
    gxf_uid_t eid_1 =kNullUid;
    gxf_uid_t* entities_uid_holder = new gxf_uid_t[1];
    uint64_t entities_count = 1024;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    const GxfEntityCreateInfo entity_create_info_1{Entity_Name_1};
    GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info_1, &eid_1));
    ASSERT_NE(eid_1,kNullUid);
    GXF_ASSERT_EQ(GxfEntityFindAll(NULL,&entities_count,entities_uid_holder), GXF_CONTEXT_INVALID);
    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFindAll,Total_Entities_equal_to_max_entities)
{
    gxf_context_t context = kNullContext;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    int Total_Entities = 1024;
    int iter=0;
    gxf_uid_t eid=kNullUid;
    std::string name="Entity_";
    std::string temp_entity_name="";
    gxf_uid_t* entities_uid_holder = new gxf_uid_t[1024];
    uint64_t entities_count = 1024;

    for(iter=1;iter<=Total_Entities;iter++)
    {
      temp_entity_name=name+std::to_string(iter);
      const char* Entity_Name=temp_entity_name.c_str();
      eid=kNullUid;
      const GxfEntityCreateInfo entity_create_info{Entity_Name};
      GXF_ASSERT_SUCCESS(GxfCreateEntity(context,&entity_create_info,&eid));
    }

    GXF_ASSERT_SUCCESS(GxfEntityFindAll(context,&entities_count, entities_uid_holder));

    for (int i=0;i<Total_Entities;i++)
     {
       GXF_ASSERT_NE(entities_uid_holder[i],0);
     }

    GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFindAll,buffer_equal_to_zero)
{
    gxf_context_t context = kNullContext;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    int Total_Entities = 1000;
    int iter=0;
    gxf_uid_t* entities_uid_holder = new gxf_uid_t[1024];
    uint64_t entities_count = 0;
    gxf_uid_t eid=kNullUid;
    std::string name="Entity_";
    std::string temp_entity_name="";
    for(iter=1;iter<=Total_Entities;iter++)
    {
       temp_entity_name=temp_entity_name+std::to_string(iter);
       const char* Entity_Name=temp_entity_name.c_str();
       eid=kNullUid;
       const GxfEntityCreateInfo entity_create_info{Entity_Name};
       ASSERT_EQ(GxfCreateEntity(context,&entity_create_info,&eid),GXF_SUCCESS);
    }

   GXF_ASSERT_EQ(GxfEntityFindAll(context, &entities_count,entities_uid_holder),GXF_QUERY_NOT_ENOUGH_CAPACITY);
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFindAll,buffer_size_equal_to_total_entities_size)
{
    gxf_context_t context = kNullContext;
    GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
    int Total_Entities = 1000;
    int iter=0;
    gxf_uid_t eid=kNullUid;
    std::string name="Entity_";
    std::string temp_entity_name="";
    gxf_uid_t* entities_uid_holder = new gxf_uid_t[1000];
    uint64_t entities_count = 1000;
    for(iter=1;iter<=Total_Entities;iter++)
    {
      temp_entity_name=name+std::to_string(iter);
      const char* Entity_Name=temp_entity_name.c_str();
      eid=kNullUid;
      const GxfEntityCreateInfo entity_create_info{Entity_Name};
      ASSERT_EQ(GxfCreateEntity(context,&entity_create_info,&eid),GXF_SUCCESS);
    }

    GXF_ASSERT_SUCCESS(GxfEntityFindAll(context, &entities_count, entities_uid_holder));
    for (int i=0;i<Total_Entities;i++)
    {
      ASSERT_NE(entities_uid_holder[i],0);
    }

   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFindAll,buffer_size_more_than_total_entities_size)
{  gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   int Total_Entities = 1000;
   int iter=0;
   gxf_uid_t eid=kNullUid;
   std::string name="Entity_";
   std::string temp_entity_name="";
   for(iter=1;iter<=Total_Entities;iter++)
   {
     temp_entity_name=name+std::to_string(iter);
     const char* Entity_Name=temp_entity_name.c_str();
     eid=kNullUid;
     const GxfEntityCreateInfo entity_create_info{Entity_Name};
     ASSERT_EQ(GxfCreateEntity(context,&entity_create_info,&eid),GXF_SUCCESS);
   }

   gxf_uid_t* entities_uid_holder = new gxf_uid_t[1000];
   uint64_t entities_count = 1200;
   gxf_result_t find_all_result = GxfEntityFindAll(context, &entities_count, entities_uid_holder);
   ASSERT_EQ(find_all_result,GXF_SUCCESS);
   for (int i=0;i<Total_Entities;i++)
   {
     ASSERT_NE(entities_uid_holder[i],0);
   }

   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFindAll,buffer_size_more_than_Total_Entities_size)
{
   gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   int Total_Entities = 1000;
   int iter=0;
   gxf_uid_t eid=kNullUid;
   std::string name="Entity_";
   std::string temp_entity_name="";
   gxf_uid_t* entities_uid_holder = new gxf_uid_t[1000];
   uint64_t entities_count = 1300;

   for(iter=1;iter<=Total_Entities;iter++)
   {
     temp_entity_name=name+std::to_string(iter);
     const char* Entity_Name=temp_entity_name.c_str();
     eid=kNullUid;
     const GxfEntityCreateInfo entity_create_info{Entity_Name};
     ASSERT_EQ(GxfCreateEntity(context,&entity_create_info,&eid),GXF_SUCCESS);
   }

   gxf_result_t find_all_result = GxfEntityFindAll(context, &entities_count,entities_uid_holder);
   ASSERT_EQ(find_all_result,GXF_SUCCESS);
   for (int i=0;i<Total_Entities;i++)
   {
     ASSERT_NE(entities_uid_holder[i],0);
   }
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}

TEST(GxfEntityFindAll,buffer_size_less_than_total_entities_size)
{  gxf_context_t context = kNullContext;
   GXF_ASSERT_SUCCESS(GxfContextCreate(&context));
   int Total_Entities = 1000;
   int iter=0;
   gxf_uid_t eid=kNullUid;
   std::string name="Entity_";
   std::string temp_entity_name="";
   gxf_uid_t* entities_uid_holder = new gxf_uid_t[1000];
   uint64_t entities_count = 900;

   for(iter=1;iter<=Total_Entities;iter++)
   {
     temp_entity_name=name+std::to_string(iter);
     const char* Entity_Name=temp_entity_name.c_str();
     eid=kNullUid;
     const GxfEntityCreateInfo entity_create_info{Entity_Name};
     ASSERT_EQ(GxfCreateEntity(context,&entity_create_info,&eid),GXF_SUCCESS);
   }

   GXF_ASSERT_EQ(GxfEntityFindAll(context, &entities_count, entities_uid_holder),GXF_QUERY_NOT_ENOUGH_CAPACITY);
   GXF_ASSERT_SUCCESS(GxfContextDestroy(context));
}