/*
Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/tensor.hpp"

#include <cinttypes>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/logger.hpp"

namespace nvidia {
namespace gxf {

namespace {
// Helper constant to compute trivial strides for tensor
constexpr Tensor::stride_array_t kTrivialStrideSteps{1, 1, 1, 1, 1, 1, 1, 1};
}  // namespace

uint64_t PrimitiveTypeSize(PrimitiveType primitive) {
  switch (primitive) {
    case PrimitiveType::kInt8:       return 1;
    case PrimitiveType::kUnsigned8:  return 1;
    case PrimitiveType::kInt16:      return 2;
    case PrimitiveType::kUnsigned16: return 2;
    case PrimitiveType::kInt32:      return 4;
    case PrimitiveType::kUnsigned32: return 4;
    case PrimitiveType::kInt64:      return 8;
    case PrimitiveType::kUnsigned64: return 8;
    case PrimitiveType::kFloat16:    return 2;
    case PrimitiveType::kFloat32:    return 4;
    case PrimitiveType::kFloat64:    return 8;
    case PrimitiveType::kComplex64:  return 8;
    case PrimitiveType::kComplex128: return 16;
    default: return 0;
  }
}

const char* primitiveTypeStr(const PrimitiveType& primitive_type) {
  switch (primitive_type) {
    GXF_ENUM_TO_STR(PrimitiveType::kCustom, kCustom)
    GXF_ENUM_TO_STR(PrimitiveType::kInt8, kInt8)
    GXF_ENUM_TO_STR(PrimitiveType::kUnsigned8, kUnsigned8)
    GXF_ENUM_TO_STR(PrimitiveType::kInt16, kInt16)
    GXF_ENUM_TO_STR(PrimitiveType::kUnsigned16, kUnsigned16)
    GXF_ENUM_TO_STR(PrimitiveType::kInt32, kInt32)
    GXF_ENUM_TO_STR(PrimitiveType::kUnsigned32, kUnsigned32)
    GXF_ENUM_TO_STR(PrimitiveType::kInt64, kInt64)
    GXF_ENUM_TO_STR(PrimitiveType::kUnsigned64, kUnsigned64)
    GXF_ENUM_TO_STR(PrimitiveType::kFloat16, kFloat16)
    GXF_ENUM_TO_STR(PrimitiveType::kFloat32, kFloat32)
    GXF_ENUM_TO_STR(PrimitiveType::kFloat64, kFloat64)
    GXF_ENUM_TO_STR(PrimitiveType::kComplex64, kComplex64)
    GXF_ENUM_TO_STR(PrimitiveType::kComplex128, kComplex128)
    default:
      return "N/A";
  }
}

const char* dlpackDeviceStr(int32_t device_type) {
  switch (device_type) {
    GXF_ENUM_TO_STR(kDLCUDA, kDLCUDA)
    GXF_ENUM_TO_STR(kDLCUDAHost, kDLCUDAHost)
    GXF_ENUM_TO_STR(kDLCPU, kDLCPU)
    GXF_ENUM_TO_STR(kDLOpenCL, kDLOpenCL)
    GXF_ENUM_TO_STR(kDLVulkan, kDLVulkan)
    GXF_ENUM_TO_STR(kDLMetal, kDLMetal)
    GXF_ENUM_TO_STR(kDLVPI, kDLVPI)
    GXF_ENUM_TO_STR(kDLROCM, kDLROCM)
    GXF_ENUM_TO_STR(kDLROCMHost, kDLROCMHost)
    GXF_ENUM_TO_STR(kDLExtDev, kDLExtDev)
    GXF_ENUM_TO_STR(kDLCUDAManaged, kDLCUDAManaged)
    GXF_ENUM_TO_STR(kDLOneAPI, kDLOneAPI)
    GXF_ENUM_TO_STR(kDLWebGPU, kDLWebGPU)
    GXF_ENUM_TO_STR(kDLHexagon, kDLHexagon)
    default:
      return "N/A";
  }
}

DLManagedMemoryBuffer::DLManagedMemoryBuffer(DLManagedTensor* self) : self_(self) {}

DLManagedMemoryBuffer::~DLManagedMemoryBuffer() {
  if (self_ && self_->deleter != nullptr) { self_->deleter(self_); }
}

Tensor::Tensor(const DLManagedTensor* dl_managed_tensor_ptr) {
  fromDLPack(dl_managed_tensor_ptr);
}

Tensor::Tensor(std::shared_ptr<DLManagedTensorContext> dl_ctx) {
  fromDLPack(dl_ctx);
}

Expected<void> Tensor::reshapeCustom(const Shape& shape,
                                     PrimitiveType element_type, uint64_t bytes_per_element,
                                     Expected<stride_array_t> strides,
                                     MemoryStorageType storage_type, Handle<Allocator> allocator) {
  if (!allocator) {
    return Unexpected{GXF_ARGUMENT_NULL};
  }

  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  shape_ = shape;
  element_count_ = shape_.size();
  element_type_ = element_type;
  bytes_per_element_ = bytes_per_element;
  strides_ = strides ? *strides : ComputeTrivialStrides(shape_, bytes_per_element_);

  result = memory_buffer_.resize(allocator, bytes_per_element * element_count_, storage_type);
  if (!result) { return ForwardError(result); }

  return initializeDLContext();
}

Expected<void> Tensor::wrapMemory(const Shape& shape,
                                  PrimitiveType element_type, uint64_t bytes_per_element,
                                  Expected<stride_array_t> strides,
                                  MemoryStorageType storage_type, void* pointer,
                                  release_function_t release_func, bool reset_dlpack) {
  shape_ = shape;
  element_count_ = shape_.size();
  element_type_ = element_type;
  bytes_per_element_ = bytes_per_element;
  strides_ = strides ? *strides : ComputeTrivialStrides(shape_, bytes_per_element_);

  auto result = memory_buffer_.wrapMemory(pointer, bytes_per_element * element_count_,
                                          storage_type, release_func);
  if (!result) { return ForwardError(result); }

  if (reset_dlpack) {
    return initializeDLContext();
  }

  return Success;
}

Expected<void> Tensor::wrapMemoryBuffer(const Shape& shape,
                                        PrimitiveType element_type, uint64_t bytes_per_element,
                                        Expected<stride_array_t> strides,
                                        MemoryBuffer memory_buffer) {
  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  shape_ = shape;
  element_count_ = shape_.size();
  element_type_ = element_type;
  bytes_per_element_ = bytes_per_element;
  strides_ = strides ? *strides : ComputeTrivialStrides(shape_, bytes_per_element_);

  memory_buffer_ = std::move(memory_buffer);

  return initializeDLContext();
}

Expected<void> Tensor::permute(const std::initializer_list<int32_t>& axes) {
  const size_t rank_tensor = rank();
  const size_t rank_permute = axes.size();
  if (rank_tensor < 2) {
    GXF_LOG_ERROR("Only tensors of rank 2 and higher can be permuted."
                  " Tensor rank is %ld", rank_tensor);
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  if (rank_tensor != rank_permute) {
    GXF_LOG_ERROR("Rank of permute input (%ld) should be the"
                  " same as rank of tensor (%ld)", rank_permute, rank_tensor);
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  std::vector<int32_t> new_dims(rank_tensor, 0);
  std::vector<uint64_t> new_strides(rank_tensor, 0);
  std::vector<bool> done(rank_tensor, false);

  int32_t i = 0;
  for (int32_t a : axes) {
    if (a >= static_cast<int32_t>(rank_tensor)) {
        GXF_LOG_ERROR("Index to permute (%d) is larger than tensor rank (%ld).",
                    a, rank_tensor);
    }
    if (done[a] == true) {
        GXF_LOG_ERROR("Cannot list the same dimension to permute twice.");
    }
    done[a] = true;
    new_dims[i] = this->shape().dimension(a);
    new_strides[i] = this->stride(a);
    i++;
  }

  // Assign new values
  shape_ = Shape(new_dims);
  std::copy_n(new_strides.begin(), rank_tensor, strides_.begin());

  // update the DLManagedTensorContext based on the new shape/strides
  return updateDLContext();
}

Expected<void> Tensor::noCopyReshape(const std::initializer_list<int32_t>& new_shape) {
  // Sanity check
  int32_t numel_new_shape = 1;
  for (int32_t d : new_shape) {numel_new_shape *= d;}
  if (static_cast<int32_t>(element_count()) != numel_new_shape) {
    GXF_LOG_ERROR("The product of the elements in new shape (%d)"
                  " must equal the product of the tensor's dimensions (%ld)",
                  numel_new_shape, element_count());
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  // Remove axes with dimension 1 from the old array. They have no effect
  // but would need special cases since their strides do not matter.
  int32_t rank_tensor = static_cast<int32_t>(rank());
  std::vector<int32_t> old_dims(rank_tensor, 0);
  Tensor::stride_array_t old_strides = kTrivialStrideSteps;
  int32_t rank_squeezed_tensor = 0;
  for (int32_t i = 0; i < rank_tensor; i++) {
    if (this->shape().dimension(i) != 1) {
      old_dims[rank_squeezed_tensor] = this->shape().dimension(i);
      old_strides[rank_squeezed_tensor] = this->stride(i);
      rank_squeezed_tensor++;
    }
  }

  // Check if new shape is compatible and if so, compute new strides
  std::vector<int32_t> new_dims(new_shape);
  int32_t num_dims_new_shape = new_shape.size();
  std::vector<uint64_t> new_strides(num_dims_new_shape, 0);
  int32_t oi = 0, ok = 0, ni = 0, nk = 0;
  int32_t oj = 1, nj = 1;
  uint64_t nd, od;
  while (ni < num_dims_new_shape && oi < rank_squeezed_tensor) {
    // Get new and old dimension
    nd = new_dims[ni];
    od = old_dims[oi];
    // Find where old and new dimensions match
    while (nd != od) {
      if (nd < od) {
        // Misses trailing 1s, handled after outer while loop breaks
        nd *= new_dims[nj++];
      } else {
        od *= old_dims[oj++];
      }
    }

    // Can the original dim be combined?
    for (ok = oi; ok < oj - 1; ok++) {
      if (old_strides[ok] != old_dims[ok + 1] * old_strides[ok + 1]) {
        GXF_LOG_ERROR("The reshape dimensions are incompatible for no-copy reshape");
        return Unexpected{GXF_ARGUMENT_NULL};
      }
    }

    // Calculate new strides for all axes currently worked with
    new_strides[nj - 1] = old_strides[oj - 1];
    for (nk = nj - 1; nk > ni; nk--) {
      new_strides[nk - 1] = new_strides[nk] * new_dims[nk];
    }

    ni = nj++;
    oi = oj++;
  }

  // Set strides corresponding to trailing 1s of the new shape.
  uint64_t last_stride;
  if (ni >= 1) {
    last_stride = new_strides[ni - 1];
  } else {
    last_stride = bytes_per_element();
  }
  for (nk = ni; nk < num_dims_new_shape; nk++) {
    new_strides[nk] = last_stride;
  }

  // Assign new shape and strides
  shape_ = Shape(new_dims);
  std::copy_n(new_strides.begin(), num_dims_new_shape, strides_.begin());

  // update the DLManagedTensorContext based on the new shape/strides
  return updateDLContext();
}

Expected<void> Tensor::insertSingletonDim(uint32_t dimension) {
  const uint32_t rank_tensor = rank();
  if (dimension < 0 || dimension > rank_tensor) {
    GXF_LOG_ERROR("Expand dimension (%d) need to be in [0, %d]",
                  dimension, rank());
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  std::vector<int32_t> new_dims(rank_tensor + 1, 1);
  std::vector<uint64_t> new_strides(rank_tensor + 1, 4);

  uint32_t j = 0;
  for (uint32_t i=0; i < rank_tensor; ++i) {
    if (dimension == i) {
      new_strides[j] = static_cast<uint64_t>(this->shape().dimension(i)) * this->stride(i);
      j++;
    }
    new_dims[j] = this->shape().dimension(i);
    new_strides[j] = this->stride(i);
    j++;
  }

  // Assign new values
  shape_ = Shape(new_dims);
  std::copy_n(new_strides.begin(), rank_tensor + 1, strides_.begin());

  // update the DLManagedTensorContext based on the new shape/strides
  return updateDLContext();
}

Expected<bool> Tensor::isContiguous() {
  // Tensor is contiguous if
  // s_i = x * \prod_{i=r - 1}^i d_i
  // for i={0, ..., r - 1}
  uint32_t r = static_cast<uint64_t>(rank());  // rank
  uint64_t x = bytes_per_element();  // total bytes
  for (int32_t i = r - 1; i >= 0; --i) {
    uint64_t s = this->stride(i);  // stride
    uint64_t d = static_cast<uint64_t>(this->shape().dimension(i));  // dimension
    if (s != x) {
      return false;
    }
    x *= d;
  }
  return true;
}

Expected<DLManagedTensor*> Tensor::toDLPack() {
  auto dl_managed_tensor_ctx = new DLManagedTensorContext;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  // call toDLManagedTensorContext() method so dl_ctx_ will be created if it is still nullptr
  auto maybe_dl_ctx = toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    ForwardError(maybe_dl_ctx);
  }
  auto dl_ctx = maybe_dl_ctx.value();
  dl_managed_tensor_ctx->memory_ref = dl_ctx->memory_ref;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx = static_cast<DLManagedTensorContext*>(self->manager_ctx);
    dl_managed_tensor_ctx->memory_ref.reset();
    delete dl_managed_tensor_ctx;
  };

  // Copy the DLTensor struct data
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor = dl_ctx->tensor.dl_tensor;
  return &dl_managed_tensor;
}

Expected<void> Tensor::wrapDLPack(const DLManagedTensor* dl_managed_tensor_ptr,
                                  MemoryBuffer::release_function_t release_func) {
  auto& dl_tensor = dl_managed_tensor_ptr->dl_tensor;
  auto maybe_shape = ShapeFromDLTensor(&dl_tensor);
  if (!maybe_shape) { return ForwardError(maybe_shape); }
  auto maybe_strides = StridesFromDLTensor(&dl_tensor);
  if (!maybe_strides) { return ForwardError(maybe_strides); }
  auto maybe_storage_type = MemoryStorageTypeFromDLTensor(&dl_tensor);
  if (!maybe_storage_type) { return ForwardError(maybe_storage_type); }
  auto maybe_element_type = PrimitiveTypeFromDLDataType(dl_tensor.dtype);
  if (!maybe_element_type) { return ForwardError(maybe_element_type); }
  auto element_type = maybe_element_type.value();
  uint64_t bytes_per_element = dl_tensor.dtype.lanes * PrimitiveTypeSize(element_type);
  // must set reset_dlpack to false when wrapping an external DLPack data structure as a Tensor
  bool reset_dlpack = false;
  auto status =
      wrapMemory(maybe_shape.value(), element_type, bytes_per_element,
                 maybe_strides.value(), maybe_storage_type.value(), dl_tensor.data, release_func,
                 reset_dlpack);
  if (!status) { ForwardError(status); }
  return Success;
}

Expected<void> Tensor::fromDLPack(const DLManagedTensor* dl_managed_tensor_ptr) {
  dl_ctx_ = std::make_shared<DLManagedTensorContext>();
  dl_ctx_->memory_ref =
      std::make_shared<DLManagedMemoryBuffer>(const_cast<DLManagedTensor*>(dl_managed_tensor_ptr));
  auto& dl_managed_tensor = dl_ctx_->tensor;
  dl_managed_tensor = *dl_managed_tensor_ptr;
  // Note: release_func argument to wrapTensor is nullptr because we use a
  // DLManagedMemoryBuffer that will handle release of the DLPack tensor memory.
  auto status = wrapDLPack(dl_managed_tensor_ptr, nullptr);
  if (!status) { ForwardError(status); }
  return Success;
}

Expected<void> Tensor::fromDLPack(std::shared_ptr<DLManagedTensorContext> dl_ctx) {
  dl_ctx_ = dl_ctx;
  // Note: release_func argument to wrapTensor is nullptr because we use a
  // DLManagedMemoryBuffer that will handle release of the DLPack tensor memory.
  auto status = wrapDLPack(&(dl_ctx->tensor), nullptr);
  if (!status) { ForwardError(status); }
  return Success;
}

Expected<void> Tensor::initializeDLContext() {
  // Get the tensor info
  const auto shape = this->shape();
  const auto element_type = this->element_type();
  const auto bytes_per_element = this->bytes_per_element();
  const auto storage_type = this->storage_type();
  const auto pointer = this->pointer();
  const auto rank = this->rank();
  const auto size = this->size();

  // Move the memory buffer from 'tensor' to 'buffer' variable with a shared pointer
  auto buffer = std::make_shared<MemoryBuffer>(std::move(this->move_buffer()));

  dl_ctx_ = std::make_shared<DLManagedTensorContext>();
  dl_ctx_->memory_ref = buffer;
  auto& dl_managed_tensor = dl_ctx_->tensor;
  auto& dl_tensor = dl_managed_tensor.dl_tensor;

  auto& buffer_shape = dl_ctx_->dl_shape;
  auto& buffer_strides = dl_ctx_->dl_strides;

  buffer_shape.reserve(rank);
  buffer_strides.reserve(rank);

  for (uint32_t index = 0; index < rank; ++index) {
    const auto stride = this->stride(index);

    buffer_shape.push_back(shape.dimension(index));
    // DLPack's stride (buffer_strides) is in elements but GXF Tensor's stride is in bytes
    buffer_strides.push_back(stride / bytes_per_element);
  }

  // change the release_func on the existing memory buffer
  auto result = memory_buffer_.wrapMemory(pointer, size, storage_type,
                                          [buffer = buffer](void* pointer) mutable {
                                            buffer.reset();
                                            return Success;
                                          });
  if (!result) { return ForwardError(result); }

  // Set the DLManagedTensorContext
  dl_managed_tensor.manager_ctx = nullptr;  // not used
  dl_managed_tensor.deleter = nullptr;      // not used

  // For Tensor, bytes_per_element may account for multiple channels (e.g. three for RGB video)
  // DLPack uses DLDataType.lanes to represent this.
  auto primitive_size = element_type == PrimitiveType::kCustom ?
                                        bytes_per_element : PrimitiveTypeSize(element_type);
  uint16_t lanes = bytes_per_element / primitive_size;
  auto maybe_dtype = PrimitiveTypeToDLDataType(element_type, lanes);
  if (!maybe_dtype) { ForwardError(maybe_dtype); }

  auto maybe_device = device();
  if (!maybe_device) { ForwardError(maybe_device); }

  dl_tensor.data = pointer;
  dl_tensor.device = maybe_device.value();
  dl_tensor.ndim = rank;
  dl_tensor.dtype = maybe_dtype.value();
  dl_tensor.shape = buffer_shape.data();
  dl_tensor.strides = buffer_strides.data();
  dl_tensor.byte_offset = 0;
  return Success;
}

Expected<DLDevice> Tensor::device() const {
  switch (storage_type()) {
    case nvidia::gxf::MemoryStorageType::kSystem:
      return DLDevice{kDLCPU, 0};
    case nvidia::gxf::MemoryStorageType::kHost:
    case nvidia::gxf::MemoryStorageType::kDevice:
      return DLDeviceFromPointer(pointer());
    default:
      GXF_LOG_ERROR("Unsupported GXF storage type (storage_type: (%d))",
                    static_cast<int>(storage_type()));
      return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
}

Expected<void> Tensor::updateDLContext() {
  // need to regenerate the DLManagedTensorContext for methods like permute or insertSingletonDim
  if (dl_ctx_ != nullptr) {
    dl_ctx_.reset();
    auto status = initializeDLContext();
    if (!status) {
      GXF_LOG_ERROR(
          "Failed to reinitialize DLManagedTensorContext with code: %s, returning nullptr",
          GxfResultStr(status.error()));
      return ForwardError(status);
    }
  }
  return Success;
}

Expected<Entity> CreateTensorMap(gxf_context_t context, Handle<Allocator> pool,
                                 std::initializer_list<TensorDescription> descriptions,
                                 bool activate) {
  Expected<Entity> message = Entity::New(context);
  if (!message) {
    return message;
  }

  // Add tensors
  std::vector<Handle<Tensor>> tensors;
  for (const TensorDescription& description : descriptions) {
    auto tensor = message.value().add<Tensor>(description.name.c_str());
    if (!tensor) {
      return ForwardError(tensor);
    }
    tensors.push_back(tensor.value());
  }

  if (activate) {
    // Activate the message entity so that we can start using it.
    const auto result = message.value().activate();
    if (!result) {
      return ForwardError(result);
    }
  }

  // Configure tensors and add them to the garbage collector
  for (size_t i = 0; i < tensors.size(); i++) {
    const TensorDescription& description = *(descriptions.begin() + i);

    const uint64_t bytes_per_element = description.element_type == PrimitiveType::kCustom
                                     ? description.bytes_per_element
                                     : PrimitiveTypeSize(description.element_type);

    auto result = tensors[i]->reshapeCustom(
        description.shape,
        description.element_type, bytes_per_element,
        description.strides,
        description.storage_type, pool);
    if (!result) {
      return ForwardError(result);
    }
  }

  return message;
}

Tensor::stride_array_t ComputeStrides(const Shape& shape,
                                      const Tensor::stride_array_t& stride_steps) {
  Tensor::stride_array_t strides;
  const uint32_t rank = shape.rank();
  if (rank == 0) {
    return strides;
  }
  strides[rank - 1] = stride_steps[rank - 1];  // lowest rank
  // Calculates proper strides from stride_steps
  for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; i--) {
    const uint64_t rank_step = stride_steps[i];
    const uint64_t dim_lower_rank = shape.dimension(i + 1);
    const uint64_t stride_lower_rank = strides[i + 1];
    strides[i] = (stride_lower_rank * dim_lower_rank + rank_step - 1) / rank_step * rank_step;
  }
  return strides;
}

Tensor::stride_array_t ComputeTrivialStrides(const Shape& shape,
                                             const uint32_t bytes_per_element) {
  Tensor::stride_array_t steps = kTrivialStrideSteps;
  const uint32_t rank = shape.rank();
  if (rank == 0) {
    return steps;
  }
  steps[rank - 1] = bytes_per_element;
  return ComputeStrides(shape, steps);
}

Expected<Tensor::stride_array_t> ComputeRowStrides(const Shape& shape, uint32_t row_step_size,
                                                   const uint32_t bytes_per_element) {
  const uint32_t rank = shape.rank();
  if (rank < 3 || row_step_size == 0) {
    return Unexpected{GXF_ARGUMENT_INVALID};
  }
  Tensor::stride_array_t steps = kTrivialStrideSteps;
  steps[rank - 3] = row_step_size;
  steps[rank - 1] = bytes_per_element;
  return ComputeStrides(shape, steps);
}

Expected<Shape> ShapeFromDLTensor(const DLTensor* dl_tensor) {
  const uint32_t rank = dl_tensor->ndim;
  if (rank > Shape::kMaxRank) {
    GXF_LOG_ERROR("Tensor rank (%d) needs to be in [0, %d]", rank, Shape::kMaxRank);
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  std::array<int32_t, Shape::kMaxRank> shape;
  for (uint32_t index = 0; index < rank; ++index) { shape[index] = dl_tensor->shape[index]; }
  return Shape(shape, rank);
}

Expected<Tensor::stride_array_t> StridesFromDLTensor(const DLTensor* dl_tensor) {
  nvidia::gxf::Tensor::stride_array_t strides;

  const uint8_t bytes_per_element = dl_tensor->dtype.bits / 8;
  // If strides is not set, set it to the default strides
  if (dl_tensor->strides == nullptr) {
    const auto shape = ShapeFromDLTensor(dl_tensor);
    if (!shape) { return ForwardError(shape); }
    strides = ComputeTrivialStrides(shape.value(), bytes_per_element);
  } else {
    const uint32_t rank = dl_tensor->ndim;
    if (rank > Shape::kMaxRank) {
      GXF_LOG_ERROR("Tensor rank (%d) needs to be in [0, %d]", rank, Shape::kMaxRank);
      return Unexpected{GXF_INVALID_DATA_FORMAT};
    }
    for (uint32_t index = 0; index < rank; ++index) {
      // GXF Tensor's stride is in bytes, but DLPack's stride is in elements
      strides[index] = dl_tensor->strides[index] * bytes_per_element;
    }
  }
  return strides;
}

Expected<MemoryStorageType> MemoryStorageTypeFromDLTensor(const DLTensor* dl_tensor) {
  const int32_t device_type = dl_tensor->device.device_type;
  MemoryStorageType storage_type = MemoryStorageType::kDevice;
  switch (device_type) {
    case kDLCUDAHost:
      storage_type = MemoryStorageType::kHost;
      break;
    case kDLCUDA:
      storage_type = MemoryStorageType::kDevice;
      break;
    case kDLCPU:
      storage_type = MemoryStorageType::kSystem;
      break;
    default:
      GXF_LOG_ERROR("Unsupported DLPack device type (%s)", dlpackDeviceStr(device_type));
      return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  return storage_type;
}

Expected<PrimitiveType> PrimitiveTypeFromDLDataType(const DLDataType& dtype) {
  PrimitiveType element_type;

  switch (dtype.code) {
    case kDLInt:
      switch (dtype.bits) {
        case 8:
          element_type = PrimitiveType::kInt8;
          break;
        case 16:
          element_type = PrimitiveType::kInt16;
          break;
        case 32:
          element_type = PrimitiveType::kInt32;
          break;
        case 64:
          element_type = PrimitiveType::kInt64;
          break;
        default:
          GXF_LOG_ERROR("Unsupported DLPack data type (code: %" PRIu8
                        ", bits: %" PRIu8 ", lanes: %" PRIu16 ")",
                        dtype.code, dtype.bits, dtype.lanes);
          return Unexpected{GXF_INVALID_DATA_FORMAT};
      }
      break;
    case kDLUInt:
      switch (dtype.bits) {
        case 8:
          element_type = PrimitiveType::kUnsigned8;
          break;
        case 16:
          element_type = PrimitiveType::kUnsigned16;
          break;
        case 32:
          element_type = PrimitiveType::kUnsigned32;
          break;
        case 64:
          element_type = PrimitiveType::kUnsigned64;
          break;
        default:
          GXF_LOG_ERROR("Unsupported DLPack data type (code: %" PRIu8
                        ", bits: %" PRIu8 ", lanes: %" PRIu16 ")",
                        dtype.code, dtype.bits, dtype.lanes);
          return Unexpected{GXF_INVALID_DATA_FORMAT};
      }
      break;
    case kDLFloat:
      switch (dtype.bits) {
        case 16:
          element_type = PrimitiveType::kFloat16;
          break;
        case 32:
          element_type = PrimitiveType::kFloat32;
          break;
        case 64:
          element_type = PrimitiveType::kFloat64;
          break;
        default:
          GXF_LOG_ERROR("Unsupported DLPack data type (code: %" PRIu8
                        ", bits: %" PRIu8 ", lanes: %" PRIu16 ")",
                        dtype.code, dtype.bits, dtype.lanes);
          return Unexpected{GXF_INVALID_DATA_FORMAT};
      }
      break;
    case kDLComplex:
      switch (dtype.bits) {
        case 64:
          element_type = PrimitiveType::kComplex64;
          break;
        case 128:
          element_type = PrimitiveType::kComplex128;
          break;
        default:
          GXF_LOG_ERROR("Unsupported DLPack data type (code: %" PRIu8
                        ", bits: %" PRIu8 ", lanes: %" PRIu16 ")",
                        dtype.code, dtype.bits, dtype.lanes);
          return Unexpected{GXF_INVALID_DATA_FORMAT};
      }
      break;
    case kDLOpaqueHandle:
    {
      element_type = PrimitiveType::kCustom;
    }
    break;
    default:
      GXF_LOG_ERROR("Unsupported DLPack data type (code: %" PRIu8
                    ", bits: %" PRIu8 ", lanes: %" PRIu16 ")",
                    dtype.code, dtype.bits, dtype.lanes);
      return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  return element_type;
}

Expected<DLDataType> PrimitiveTypeToDLDataType(const PrimitiveType& element_type, uint16_t lanes) {
  if (lanes < 1) {
    GXF_LOG_ERROR("Lanes must be a positive integer, found (%" PRIu16 ")", lanes);
    return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  DLDataType dtype;
  dtype.lanes = lanes;
  dtype.bits = PrimitiveTypeSize(element_type) * 8;
  switch (element_type) {
    case PrimitiveType::kInt8:
    case PrimitiveType::kInt16:
    case PrimitiveType::kInt32:
    case PrimitiveType::kInt64:
      dtype.code = kDLInt;
      break;
    case PrimitiveType::kUnsigned8:
    case PrimitiveType::kUnsigned16:
    case PrimitiveType::kUnsigned32:
    case PrimitiveType::kUnsigned64:
      dtype.code = kDLUInt;
      break;
    case PrimitiveType::kFloat16:
    case PrimitiveType::kFloat32:
    case PrimitiveType::kFloat64:
      dtype.code = kDLFloat;
      break;
    case PrimitiveType::kComplex64:
    case PrimitiveType::kComplex128:
      dtype.code = kDLComplex;
      break;
    case PrimitiveType::kCustom:
      dtype.code = kDLOpaqueHandle;
      break;
    default:
      GXF_LOG_ERROR("Unsupported primitive type (%s)", primitiveTypeStr(element_type));
      return Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  return dtype;
}
}  // namespace gxf
}  // namespace nvidia
