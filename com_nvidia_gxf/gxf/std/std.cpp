/*
Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/block_memory_pool.hpp"
#include "gxf/std/broadcast.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/complex.hpp"
#include "gxf/std/connection.hpp"
#include "gxf/std/controller.hpp"
#include "gxf/std/cpu_thread.hpp"
#include "gxf/std/double_buffer_receiver.hpp"
#include "gxf/std/double_buffer_transmitter.hpp"
#include "gxf/std/entity_executor.hpp"
#include "gxf/std/eos.hpp"
#include "gxf/std/epoch_scheduler.hpp"
#include "gxf/std/extension_factory_helper.hpp"
#include "gxf/std/forward.hpp"
#include "gxf/std/gather.hpp"
#include "gxf/std/greedy_scheduler.hpp"
#include "gxf/std/ipc_server.hpp"
#include "gxf/std/job_statistics.hpp"
#include "gxf/std/message_router.hpp"
#include "gxf/std/metric.hpp"
#include "gxf/std/monitor.hpp"
#include "gxf/std/multi_thread_scheduler.hpp"
#include "gxf/std/network_context.hpp"
#include "gxf/std/network_router.hpp"
#include "gxf/std/queue.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/resources.hpp"
#include "gxf/std/router.hpp"
#include "gxf/std/router_group.hpp"
#include "gxf/std/scheduler.hpp"
#include "gxf/std/scheduling_term.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/subgraph.hpp"
#include "gxf/std/synchronization.hpp"
#include "gxf/std/synthetic_clock.hpp"
#include "gxf/std/system.hpp"
#include "gxf/std/system_group.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/tensor_copier.hpp"
#include "gxf/std/timed_throttler.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/transmitter.hpp"
#include "gxf/std/unbounded_allocator.hpp"
#include "gxf/std/vault.hpp"

GXF_EXT_FACTORY_BEGIN()
  GXF_EXT_FACTORY_SET_INFO(0x8ec2d5d6b5df48bf, 0x8dee0252606fdd7e, "StandardExtension",
                           "Most commonly used interfaces and components in Gxf Core",
                           "NVIDIA", "2.3.0", "NVIDIA");
  GXF_EXT_FACTORY_SET_DISPLAY_INFO("Standard Extension", "Standard", "GXF Standard Extension");
  GXF_EXT_FACTORY_ADD(0x5c6166fa6eed41e7, 0xbbf0bd48cd6e1014,
                      nvidia::gxf::Codelet, nvidia::gxf::Component,
                      "Interface for a component which can be executed to run custom code");
  GXF_EXT_FACTORY_ADD(0x00e6f23d0bf64c1c, 0xada5630c711d3be1,
                      nvidia::gxf::IPCServer, nvidia::gxf::Component,
                      "Interface for a component which works as a API server to respond on "
                      "remote requests");
  GXF_EXT_FACTORY_ADD(0xd589ce20f3a74fc0, 0xaa4ed6f40f6c0eeb,
                      nvidia::gxf::NetworkContext, nvidia::gxf::Component,
                      "Interface for a component for network context like UCX ");
  GXF_EXT_FACTORY_ADD(0x779e61c2ae70441d, 0xa26c8ca64b39f8e7,
                      nvidia::gxf::Clock, nvidia::gxf::Component,
                      "Interface for clock components which provide time");
  GXF_EXT_FACTORY_ADD(0x7b170b7bcf1a4f3f, 0x997cbfea25342381,
                      nvidia::gxf::RealtimeClock, nvidia::gxf::Clock,
                      "A realtime clock which runs based off a system steady clock");
  GXF_EXT_FACTORY_ADD(0x52fa1f97eba8472a, 0xa8ca4cff1a2c440f,
                      nvidia::gxf::ManualClock, nvidia::gxf::Clock,
                      "A manual clock which is instrumented manually");
  GXF_EXT_FACTORY_ADD(0xd1febca180df454e, 0xa3f2715f2b3c6e69,
                      nvidia::gxf::System, nvidia::gxf::Component,
                      "Component interface for systems which are run as part of the "
                      "application run cycle");
  GXF_EXT_FACTORY_ADD(0x3d23d4700aed41c6, 0xac92685c1b5469a0,
                      nvidia::gxf::SystemGroup, nvidia::gxf::System,
                      "A group of systems");
  GXF_EXT_FACTORY_ADD(0x792151bf31384603, 0xa9125ca91828dea8,
                      nvidia::gxf::Queue, nvidia::gxf::Component,
                      "Interface for storing entities in a queue");

  GXF_EXT_FACTORY_ADD_0(0x8b317aadf55c4c07, 0x85208f66db92a19e,
                      nvidia::gxf::Router,
                      "Interface for classes which are routing messages in and out of entities.");
  GXF_EXT_FACTORY_ADD(0x84fd5d56fda64937, 0xb3cc283252553d8,
                      nvidia::gxf::MessageRouter, nvidia::gxf::Router,
                      "A router which sends transmitted messages to receivers.");
  GXF_EXT_FACTORY_ADD(0xa1e6c5d5947d40fd, 0xb248117dddc9f0ae,
                      nvidia::gxf::NetworkRouter, nvidia::gxf::Router,
                      "A router which sends transmitted messages to remote receivers.");
  GXF_EXT_FACTORY_ADD(0xca64ee1422804099, 0x9f10d4b501e09117,
                      nvidia::gxf::RouterGroup, nvidia::gxf::Router,
                      "A group of routers");

  GXF_EXT_FACTORY_ADD(0xc30cc60f0db2409d, 0x92b6b2db92e02cce,
                      nvidia::gxf::Transmitter, nvidia::gxf::Queue,
                      "Interface for publishing entities");
  GXF_EXT_FACTORY_ADD(0xa47d2f62245f40fc, 0x90b75dc78ff2437e,
                      nvidia::gxf::Receiver, nvidia::gxf::Queue,
                      "Interface for receiving entities");
  GXF_EXT_FACTORY_ADD(0x0c3c0ec777f14389, 0xaef16bae85bddc13,
                      nvidia::gxf::DoubleBufferTransmitter, nvidia::gxf::Transmitter,
                      "A transmitter which uses a double-buffered queue where messages"
                      " are pushed to a backstage after they are published");
  GXF_EXT_FACTORY_ADD(0xee45883dbf844f99, 0x84197c5e9deac6a5,
                      nvidia::gxf::DoubleBufferReceiver, nvidia::gxf::Receiver,
                      "A receiver which uses a double-buffered queue where new "
                      "messages are first pushed to a backstage");
  GXF_EXT_FACTORY_ADD(0xcc71afae5ede47e9, 0xb26760a5c750a89a,
                      nvidia::gxf::Connection, nvidia::gxf::Component,
                      "A component which establishes a connection between two other components");
  GXF_EXT_FACTORY_ADD(0x76b9234d5eac4c65, 0xb1a10306d3f354e5,
                      nvidia::gxf::ResourceBase, nvidia::gxf::Component,
                      "A Resource base type");
  GXF_EXT_FACTORY_ADD(0x4025b68b206b4b3d, 0xa088f4805fdf8703,
                      nvidia::gxf::ThreadPool, nvidia::gxf::ResourceBase,
                      "A threadpool component we can use to prioritize workloads");
  GXF_EXT_FACTORY_ADD(0x2036939fa32a43ee, 0x83f8826576d8f170,
                      nvidia::gxf::GPUDevice, nvidia::gxf::ResourceBase,
                      "A GPU Resource for codelet workloads");
  GXF_EXT_FACTORY_ADD(0x3cdd82d023264867, 0x8de2d565dbe28e03,
                      nvidia::gxf::Allocator, nvidia::gxf::Component,
                      "Provides allocation and deallocation of memory");
  GXF_EXT_FACTORY_ADD(0x92b627a35dd34c3c, 0x976c4700e8a3b96a,
                      nvidia::gxf::BlockMemoryPool, nvidia::gxf::Allocator,
                      "A memory pools which provides a maximum number of equally sized "
                      "blocks of memory");
  GXF_EXT_FACTORY_ADD(0xc3951b16a01c539f, 0xd87e1dc18d911ea0,
                      nvidia::gxf::UnboundedAllocator, nvidia::gxf::Allocator,
                      "Allocator that uses dynamic memory allocation without an upper bound");
  GXF_EXT_FACTORY_ADD(0xf0103b75d2e14d70, 0x9b133fe5b40209be,
                      nvidia::gxf::Scheduler, nvidia::gxf::System,
                      "A simple poll-based single-threaded scheduler which executes codelets");
  GXF_EXT_FACTORY_ADD(0x184d8e4e086c475a, 0x903a69d723f95d19,
                      nvidia::gxf::SchedulingTerm, nvidia::gxf::Component,
                      "Interface for terms used by a scheduler to determine if codelets in"
                      " an entity are ready to step");
  GXF_EXT_FACTORY_ADD(0xd392c98a9b0849b4, 0xa422d5fe6cd72e3e,
                      nvidia::gxf::PeriodicSchedulingTerm, nvidia::gxf::SchedulingTerm,
                      "A component which specifies that an entity shall be executed periodically");
  GXF_EXT_FACTORY_ADD(0xf89da2e4fddf4aa2, 0x9a801119ba3fde05,
                      nvidia::gxf::CountSchedulingTerm, nvidia::gxf::SchedulingTerm,
                      "A component which specifies that an entity shall be executed"
                      " exactly a given number of times");
  GXF_EXT_FACTORY_ADD(0xe4aaf5c32b104c9a, 0xc463ebf6084149bf,
                    nvidia::gxf::TargetTimeSchedulingTerm, nvidia::gxf::SchedulingTerm,
                    "A component where the next execution time of the entity needs"
                    "to be specified after every tick");
  GXF_EXT_FACTORY_ADD(0x9de751198d0f4819, 0x9a712aeaefd23f71,
                      nvidia::gxf::DownstreamReceptiveSchedulingTerm, nvidia::gxf::SchedulingTerm,
                      "A component which specifies that an entity shall be executed if receivers "
                      "for a certain transmitter can accept new messages");
  GXF_EXT_FACTORY_ADD(0xfe799e65f78b48eb, 0xbeb6e73083a12d5b,
                      nvidia::gxf::MessageAvailableSchedulingTerm, nvidia::gxf::SchedulingTerm,
                      "A scheduling term which specifies that an entity can be executed when the "
                      "total number of messages over a set of input channels is at least a given "
                      "number of messages.");
  GXF_EXT_FACTORY_ADD(0xf15dbeaaafd647a6, 0x9ffc7afd7e1b4c52,
                      nvidia::gxf::MultiMessageAvailableSchedulingTerm, nvidia::gxf::SchedulingTerm,
                      "A component which specifies that an entity shall be executed when a queue"
                      " has at least a certain number of elements");
  GXF_EXT_FACTORY_ADD(0xeb22280c76ff11eb, 0xb341cf6b417c95c9,
                      nvidia::gxf::ExpiringMessageAvailableSchedulingTerm,
                      nvidia::gxf::SchedulingTerm,
                      "A component which tries to wait for specified number of messages in queue"
                      " for at most specified time.");
  GXF_EXT_FACTORY_ADD(0xe07a0dc439084df8, 0x81347ce38e60fbef,
                      nvidia::gxf::BooleanSchedulingTerm,
                      nvidia::gxf::SchedulingTerm,
                      "A component which acts as a boolean AND term that can be used to control the"
                      " execution of the entity.");
  GXF_EXT_FACTORY_ADD(0x0161ca512fed4a8c, 0x8f2467cf1b5e330a,
                      nvidia::gxf::BTSchedulingTerm, nvidia::gxf::SchedulingTerm,
                      "A component which is used to control the"
                      " execution of the behavior tree entities.");
  GXF_EXT_FACTORY_ADD(0x56be1662ff634179, 0x92003fcd8dc38673,
                      nvidia::gxf::AsynchronousSchedulingTerm,
                      nvidia::gxf::SchedulingTerm,
                      "A component which is used to inform of that an entity is dependent upon an"
                      " async event for its execution");
  GXF_EXT_FACTORY_ADD(0x5ae1d56ca19611ed, 0x8759ef34a33d45a6,
                      nvidia::gxf::MessageAvailableFrequencyThrottler,
                      nvidia::gxf::SchedulingTerm,
                      "A component which is used to execute an entity at specific frequency or"
                      " sooner if there are a minimum number of incoming messages in its "
                      "receivers");
  GXF_EXT_FACTORY_ADD(0xf976d23a822074e2, 0xa5d904ed71b2454a,
                      nvidia::gxf::MemoryAvailableSchedulingTerm,
                      nvidia::gxf::SchedulingTerm,
                      "A component waiting until a minimum amount of memory is available");
  GXF_EXT_FACTORY_ADD(0x869d30caa4434619, 0xb9887a52e657f39b,
                      nvidia::gxf::GreedyScheduler, nvidia::gxf::Scheduler,
                      "A simple poll-based single-threaded scheduler which executes codelets");
  GXF_EXT_FACTORY_ADD(0xde5e06467fa511eb, 0xa5c4330ebfa81bbf,
                      nvidia::gxf::MultiThreadScheduler, nvidia::gxf::Scheduler,
                      "A multi thread scheduler that executes codelets for maximum throughput.");
  GXF_EXT_FACTORY_ADD_0(0x377501d69abf447c, 0xa6170114d4f33ab8,
                        nvidia::gxf::Tensor,
                        "A component which holds a single tensor");
  GXF_EXT_FACTORY_ADD_0(0x469cc839a6ab470b, 0x9d09b8bf978c13cd,
                        nvidia::gxf::Shape,
                        "A component which describes the shape of a tensor");
  GXF_EXT_FACTORY_ADD_0(0xd1095b105c904bbc, 0xbc89601134cb4e03,
                        nvidia::gxf::Timestamp,
                        "Holds message publishing and acquisition related timing information");
  GXF_EXT_FACTORY_ADD_0(0x872e77ecbde811ed, 0xafa10242ac120002,
                        nvidia::gxf::MultiSourceTimestamp,
                        "Holds timestamps from various sources");
  GXF_EXT_FACTORY_ADD(0xf7cef8035beb46f1, 0x186a05d3919842ac,
                      nvidia::gxf::Metric, nvidia::gxf::Component,
                     "Collects, aggregates, and evaluates metric data.");
  GXF_EXT_FACTORY_ADD(0x2093b91a7c8211eb, 0xa92b3f1304ecc959,
                      nvidia::gxf::JobStatistics, nvidia::gxf::Component,
                      "Collects runtime statistics.");
  GXF_EXT_FACTORY_ADD(0x9ccf9421b35b8c79, 0xe1f097dc23bd38ea,
                      nvidia::gxf::Monitor, nvidia::gxf::Component,
                      "Monitors entities during execution.");
  GXF_EXT_FACTORY_ADD(0xc8e804753c7943a4, 0x9083eaf294b0600d,
                      nvidia::gxf::Controller, nvidia::gxf::Component,
                      "Controls entities' termination policy and tracks behavior status"
                      " during execution.");
  GXF_EXT_FACTORY_ADD(0x3daadb310bca47e5, 0x9924342b9984a014,
                      nvidia::gxf::Broadcast, nvidia::gxf::Codelet,
                      "Messages arrived on the input channel are distributed to all transmitters.");
  GXF_EXT_FACTORY_ADD(0x85f64c8482364035, 0x9b9a3843a6a2026f,
                      nvidia::gxf::Gather, nvidia::gxf::Codelet,
                      "All messages arriving on any input channel are published on the single "
                      "output channel.");
  GXF_EXT_FACTORY_ADD(0xc07680f475b3189b, 0x88864b5e448e7bb6,
                      nvidia::gxf::TensorCopier, nvidia::gxf::Codelet,
                      "Copies tensor either from host to device or from device to host");
  GXF_EXT_FACTORY_ADD(0xccf7729cf62c4250, 0x5cf7f4f3ec80454b,
                      nvidia::gxf::TimedThrottler, nvidia::gxf::Codelet,
                      "Publishes the received entity respecting the timestamp within the entity");
  GXF_EXT_FACTORY_ADD(0x1108cb8d85e44303, 0xba02d27406ee9e65,
                      nvidia::gxf::Vault, nvidia::gxf::Codelet,
                      "Safely stores received entities for further processing.");
  GXF_EXT_FACTORY_ADD(0x576eedd77c3f4d2f, 0x8c388baa79a3d231,
                      nvidia::gxf::Subgraph, nvidia::gxf::Component,
                      "Helper component to import a subgraph");
  GXF_EXT_FACTORY_ADD_0(0x8c42f7bf70414626, 0x97929eb20ce33cce,
                      nvidia::gxf::EndOfStream,
                      "A component which represents end-of-stream notification");
  GXF_EXT_FACTORY_ADD(0xf1cb80d6e5ec4dba, 0x9f9eb06b0def4443,
                      nvidia::gxf::Synchronization, nvidia::gxf::Codelet,
                      "Component to synchronize messages from multiple receivers based on the"
                      "acq_time");
  GXF_EXT_FACTORY_ADD(0x3d175ab42e0d11ec, 0x8d3d0242ac130003,
                      nvidia::gxf::EpochScheduler, nvidia::gxf::Scheduler,
                      "A scheduler for running loads in externally managed threads");
  GXF_EXT_FACTORY_ADD(0x34f46728496d4d8b, 0xb9c9c5a54de5d3a0,
                      nvidia::gxf::CPUThread, nvidia::gxf::Component,
                      "A resource component used to pin jobs to a given thread.");
  GXF_EXT_FACTORY_ADD(0x9a2bfd7b2d8479b4, 0xbc71f47eb53f28c8,
                      nvidia::gxf::SyntheticClock, nvidia::gxf::Clock,
                      "A synthetic clock used to inject simulated time");
  GXF_EXT_FACTORY_ADD(0x97cee5438fb54541, 0x8ff7589318187ec0,
                       nvidia::gxf::Forward, nvidia::gxf::Codelet,
                       "Forwards incoming messages at the receiver to the transmitter");

  GXF_EXT_FACTORY_ADD_0_LITE(0x83905c6aca344f40, 0xb474cf2cde8274de, int8_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0xd4299e150006d0bf, 0x8cbd9b743575e155, uint8_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0x9e1dde793550307d, 0xe81ab864890b3685, int16_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0x958cbdefb505bcc7, 0x8a43dc4b23f8cead, uint16_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0xb557ec7f49a508f7, 0xa35e086e9d1ea767, int32_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0xd5506b685c86fedb, 0xa2a2a7bae38ff3ef, uint32_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0xc611627b6393365f, 0xd2341f26bfa8d28f, int64_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0xc4385f5b6e2501d9, 0xd7b56e7cadc704e8, uint64_t);
  GXF_EXT_FACTORY_ADD_0_LITE(0xa81bf295421f49ef, 0xf24af59e9ea0d5d6, float);
  GXF_EXT_FACTORY_ADD_0_LITE(0xd57cee59686fe26d, 0x95be659c126b02ea, double);
  GXF_EXT_FACTORY_ADD_0_LITE(0xc02f9e93d01b1d29, 0xf52378d2a9195128, bool);
  GXF_EXT_FACTORY_ADD_0_LITE(0x97edf9034ea54e59, 0x8a44de9754d40b7f, nvidia::gxf::complex64);
  GXF_EXT_FACTORY_ADD_0_LITE(0x0a401e6d4ca74d21, 0xa6a0fe6a2fcc37ea, nvidia::gxf::complex128);
GXF_EXT_FACTORY_END()
