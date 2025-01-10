/*
Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <utility>

#include "gxf/stream/stream_nvscisync.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t StreamSync::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      signaler_, "signaler", "Signaler type",
      "Defines the signaler type. Cuda Signaler (0)",
      static_cast<int32_t>(SyncType::GXF_STREAM_SIGNALER_CUDA));
  result &= registrar->parameter(
      waiter_, "waiter", "Waiter type",
      "Defines the waiter type. Cuda Signaler (0)",
      static_cast<int32_t>(SyncType::GXF_STREAM_WAITER_CUDA));
  result &= registrar->parameter(
      signaler_device_id_, "signaler_gpu_id", "Signaler GPU Id", "Cuda device ID for Signaler", 0);
  result &= registrar->parameter(
      waiter_device_id_, "waiter_gpu_id", "Waiter GPU Id", "Cuda device ID for Waiter", 0);
  return ToResultCode(result);
}

gxf_result_t StreamSync::initialize() {
  NvSciError error = NvSciError_Success;

  if ((static_cast<SyncType>(signaler_.get()) != SyncType::GXF_STREAM_SIGNALER_CUDA) ||
      (static_cast<SyncType>(waiter_.get()) != SyncType::GXF_STREAM_WAITER_CUDA)) {
    GXF_LOG_ERROR("GXF Stream sync not supported");
    return GXF_FAILURE;
  }

  error = NvSciSyncModuleOpen(&sync_module_);
  if (error != NvSciError_Success) {
    GXF_LOG_ERROR("NvSciSyncModuleOpen Failed - e = %d", error);
    return GXF_FAILURE;
  }

  fence_ = static_cast<NvSciSyncFence *>(calloc(1, sizeof(NvSciSyncFence)));

  cudaError_t cudaError = cudaGetDeviceCount(&num_gpus_);
  if (cudaError != cudaSuccess) {
      GXF_LOG_ERROR("cudaGetDeviceCount Failed - %s", cudaGetErrorString(cudaError));
      return GXF_FAILURE;
  }

  if (signaler_device_id_.get() >= num_gpus_) {
    GXF_LOG_ERROR("Device ID for signaler is greater than the available device count - signaler_device_id = %d, device_count = %d", signaler_device_id_.get(), num_gpus_); // NOLINT
    return GXF_FAILURE;
  }

  if (waiter_device_id_.get() >= num_gpus_) {
    GXF_LOG_ERROR("Device ID for waiter is greater than the available device count - waiter_device_id = %d, device_count = %d", waiter_device_id_.get(), num_gpus_); // NOLINT
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t StreamSync::deinitialize() {
  delete fence_;
  if (reconciled_attr_list_) {
    NvSciSyncAttrListFree(reconciled_attr_list_);
  }
  if (sync_obj_) {
    NvSciSyncObjFree(sync_obj_);
  }
  if (sync_module_) {
    NvSciSyncModuleClose(sync_module_);
  }
  return GXF_SUCCESS;
}

gxf_result_t StreamSync::allocate_sync_object(SyncType signaler, SyncType waiter, void** syncObj) {
  NvSciSyncAttrList signaler_attr_list{nullptr};
  NvSciSyncAttrList waiter_attr_list{nullptr};
  NvSciSyncAttrList unreconciled_list[2]{nullptr};
  NvSciSyncAttrList conflict_attr_list{nullptr};
  NvSciError error{NvSciError_Success};
  cudaError_t cuda_error{cudaSuccess};

  if (static_cast<SyncType>(signaler_.get()) != signaler) {
    GXF_LOG_ERROR("Incorrect signaler type - Provided: %d, Expected: %d",
                  static_cast<int32_t>(signaler), signaler_.get());
    return GXF_ARGUMENT_INVALID;
  }

  if (static_cast<SyncType>(waiter_.get()) != waiter) {
    GXF_LOG_ERROR("Incorrect waiter type - Provided: %d, Expected: %d",
                  static_cast<int32_t>(waiter), waiter_.get());
    return GXF_ARGUMENT_INVALID;
  }

  error = NvSciSyncAttrListCreate(sync_module_, &signaler_attr_list);
  if (error != NvSciError_Success) {
    GXF_LOG_ERROR("NvSciSyncAttrListCreate Failed for signaler - e = %d",
                  static_cast<int32_t>(error));
    return GXF_FAILURE;
  }

  error = NvSciSyncAttrListCreate(sync_module_, &waiter_attr_list);
  if (error != NvSciError_Success) {
    GXF_LOG_ERROR("NvSciSyncAttrListCreate Failed for waiter - e = %d",
                  static_cast<int32_t>(error));
    return GXF_FAILURE;
  }

  if (signaler == SyncType::GXF_STREAM_SIGNALER_CUDA) {
    cuda_error = cudaDeviceGetNvSciSyncAttributes(signaler_attr_list, signaler_device_id_,
                                                  cudaNvSciSyncAttrSignal);
    if (cuda_error != cudaSuccess) {
      GXF_LOG_ERROR("cudaDeviceGetNvSciSyncAttributes for signaler Failed - %s",
                    cudaGetErrorString(cuda_error));
      return GXF_FAILURE;
    }
  }

  if (waiter == SyncType::GXF_STREAM_WAITER_CUDA) {
    cuda_error = cudaDeviceGetNvSciSyncAttributes(waiter_attr_list, waiter_device_id_,
                                                  cudaNvSciSyncAttrWait);
    if (cuda_error != cudaSuccess) {
      GXF_LOG_ERROR("cudaDeviceGetNvSciSyncAttributes for waiter Failed - %s",
                    cudaGetErrorString(cuda_error));
      return GXF_FAILURE;
    }
  }

  unreconciled_list[0] = signaler_attr_list;
  unreconciled_list[1] = waiter_attr_list;
  error = NvSciSyncAttrListReconcile(unreconciled_list, 2,
                                     &reconciled_attr_list_, &conflict_attr_list);
  if (error != NvSciError_Success) {
    GXF_LOG_ERROR("NvSciSyncAttrListReconcile Failed - e = %d", static_cast<int32_t>(error));
    return GXF_FAILURE;
  }

  error = NvSciSyncObjAlloc(reconciled_attr_list_, &sync_obj_);
  if (error != NvSciError_Success) {
    GXF_LOG_ERROR("NvSciSyncObjAlloc Failed - e = %d", static_cast<int32_t>(error));
    return GXF_FAILURE;
  }

  if (signaler_attr_list != nullptr) {
    NvSciSyncAttrListFree(signaler_attr_list);
  }
  if (waiter_attr_list != nullptr) {
    NvSciSyncAttrListFree(waiter_attr_list);
  }
  if (conflict_attr_list != nullptr) {
    NvSciSyncAttrListFree(conflict_attr_list);
  }

  *syncObj = sync_obj_;

  return GXF_SUCCESS;
}

gxf_result_t StreamSync::setCudaStream(SyncType syncType, cudaStream_t stream) {
  if (stream == NULL) {
    GXF_LOG_ERROR("Invalid cuda stream");
    return GXF_ARGUMENT_INVALID;
  }

  if (syncType == SyncType::GXF_STREAM_SIGNALER_CUDA) {
    signaler_cuda_stream_ = std::move(stream);
  } else if (syncType == SyncType::GXF_STREAM_WAITER_CUDA) {
    waiter_cuda_stream_ = std::move(stream);
  } else {
    GXF_LOG_ERROR("Invalid syncType = %d", static_cast<int32_t>(syncType));
    return GXF_ARGUMENT_INVALID;
  }

  return GXF_SUCCESS;
}

gxf_result_t StreamSync::signalSemaphore() {
  if (sync_obj_ == NULL) {
    GXF_LOG_ERROR("Sync object is not initialized");
    return GXF_ARGUMENT_INVALID;
  }

  SyncType signaler = static_cast<SyncType>(signaler_.get());
  importSemaphore(signaler);

  if (is_signaler_semaphore_imported_ == false) {
    GXF_LOG_ERROR("Signaler semaphore is not imported. Import semaphore before calling signal on semaphore"); // NOLINT
    return GXF_FAILURE;
  }
  if (signaler == SyncType::GXF_STREAM_SIGNALER_CUDA) {
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = static_cast<void *>(fence_);
    signalParams.flags = 0;
    cudaError_t error = cudaSignalExternalSemaphoresAsync(&signaler_semaphore_,
                                                          &signalParams, 1, signaler_cuda_stream_);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("cudaSignalExternalSemaphoresAsync Failed - %s", cudaGetErrorString(error));
      return GXF_FAILURE;
    }
  } else {
    GXF_LOG_ERROR("Unknown signaler type - %d", static_cast<int32_t>(signaler));
    return GXF_ARGUMENT_INVALID;
  }
  return GXF_SUCCESS;
}

gxf_result_t StreamSync::importSemaphore(SyncType syncType) {
  gxf_result_t result{GXF_SUCCESS};
  if (sync_obj_ == NULL) {
    GXF_LOG_ERROR("Sync object is not initialized");
    return GXF_ARGUMENT_INVALID;
  }
  if (syncType == SyncType::GXF_STREAM_SIGNALER_CUDA) {
    result = importSemaphore(&signaler_semaphore_, syncType);
    if (result == GXF_SUCCESS) {
      is_signaler_semaphore_imported_ = true;
    } else {
      return result;
    }
  } else if (syncType == SyncType::GXF_STREAM_WAITER_CUDA) {
    result = importSemaphore(&waiter_semaphore_, syncType);
    if (result == GXF_SUCCESS) {
      is_waiter_semaphore_imported_ = true;
    } else {
      return result;
    }
  } else {
    GXF_LOG_ERROR("Cannot import semaphore for sync type %d", static_cast<int32_t>(syncType));
    return GXF_ARGUMENT_INVALID;
  }
  return result;
}

gxf_result_t StreamSync::importSemaphore(cudaExternalSemaphore_t* semaphore,
                                              SyncType syncType) {
  if (sync_obj_ == NULL) {
    GXF_LOG_ERROR("Sync object is not initialized");
    return GXF_ARGUMENT_INVALID;
  }
  cudaError_t error{cudaSuccess};
  if (syncType == SyncType::GXF_STREAM_SIGNALER_CUDA) {
    error = cudaSetDevice(signaler_device_id_.get());
  } else if (syncType == SyncType::GXF_STREAM_WAITER_CUDA) {
    error = cudaSetDevice(waiter_device_id_.get());
  } else {
    GXF_LOG_ERROR("Cannot setDevice for unknown sync type - %d", static_cast<int32_t>(syncType));
    return GXF_ARGUMENT_INVALID;
  }
  if (error != cudaSuccess) {
    GXF_LOG_ERROR("cudaSetDevice Failed - %s", cudaGetErrorString(error));
    return GXF_FAILURE;
  }

  cudaExternalSemaphoreHandleDesc extSemDesc;
  memset(&extSemDesc, 0, sizeof(extSemDesc));
  extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
  extSemDesc.handle.nvSciSyncObj = sync_obj_;

  error = cudaImportExternalSemaphore(semaphore, &extSemDesc);
  if (error != cudaSuccess) {
    GXF_LOG_ERROR("cudaImportExternalSemaphore Failed - %s", cudaGetErrorString(error));
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t StreamSync::waitSemaphore() {
  if (sync_obj_ == NULL) {
    GXF_LOG_ERROR("Sync object is not initialized");
    return GXF_ARGUMENT_INVALID;
  }

  SyncType waiter = static_cast<SyncType>(waiter_.get());

  importSemaphore(waiter);
  if (is_waiter_semaphore_imported_ == false) {
    GXF_LOG_ERROR(" Wait semaphore is not imported. Import semaphore before calling wait on semaphore"); //NOLINT
    return GXF_FAILURE;
  }

  if (waiter == SyncType::GXF_STREAM_WAITER_CUDA) {
    cudaError_t error = cudaSetDevice(waiter_device_id_.get());
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("cudaSetDevice Failed - %s", cudaGetErrorString(error));
    }

    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    waitParams.params.nvSciSync.fence = static_cast<void *>(fence_);
    waitParams.flags = 0;;
    waitParams.flags = 0;

    error = cudaWaitExternalSemaphoresAsync(&waiter_semaphore_, &waitParams,
                                            1, waiter_cuda_stream_);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("cudaWaitExternalSemaphoresAsync Failed - %s", cudaGetErrorString(error));
      return GXF_FAILURE;
    }
  } else {
    GXF_LOG_ERROR("Unknown waiter type - %d", static_cast<int32_t>(waiter));
    return GXF_ARGUMENT_INVALID;
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
