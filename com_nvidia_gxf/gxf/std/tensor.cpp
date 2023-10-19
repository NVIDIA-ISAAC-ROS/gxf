/*
Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gxf/std/tensor.hpp"

#include <utility>
#include <vector>

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
    case PrimitiveType::kFloat32:    return 4;
    case PrimitiveType::kFloat64:    return 8;
    case PrimitiveType::kComplex64:  return 8;
    case PrimitiveType::kComplex128: return 16;
    default: return 0;
  }
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
  return Success;
}

Expected<void> Tensor::wrapMemory(const Shape& shape,
                                  PrimitiveType element_type, uint64_t bytes_per_element,
                                  Expected<stride_array_t> strides,
                                  MemoryStorageType storage_type, void* pointer,
                                  release_function_t release_func) {
  auto result = memory_buffer_.freeBuffer();
  if (!result) { return ForwardError(result); }

  shape_ = shape;
  element_count_ = shape_.size();
  element_type_ = element_type;
  bytes_per_element_ = bytes_per_element;
  strides_ = strides ? *strides : ComputeTrivialStrides(shape_, bytes_per_element_);

  result = memory_buffer_.wrapMemory(pointer, bytes_per_element * element_count_,
                                     storage_type, release_func);
  if (!result) { return ForwardError(result); }
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
  return Success;
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

  return Success;
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
  std::copy_n(new_strides.begin(), rank_tensor, strides_.begin());

  return Success;
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

  return Success;
}

Expected<bool> Tensor::isContiguous() {
  // Tensor is contiguous if
  // s_i = x * \prod_{i=r - 1}^i d_i
  // for i={0, ..., r - 1}
  uint32_t r = static_cast<uint64_t>(rank());  // rank
  uint64_t x = bytes_per_element();  // total bytes
  for (uint32_t i = r - 1; i >= 0; --i) {
    uint64_t s = this->stride(i);  // stride
    uint64_t d = static_cast<uint64_t>(this->shape().dimension(i));  // dimension
    if (s != x) {
      return false;
    }
    x *= d;
  }
  return true;
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

}  // namespace gxf
}  // namespace nvidia
