/*
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/logger.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/dlpack_utils.hpp"
#include "gxf/std/tensor.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace {

// This macros is like CHECK_CUDA_ERROR from gxf/cuda/cuda_common.h, but it throws a runtime_error
// instead of returning Unexpected
#define CHECK_CUDA_THROW_ERROR(cu_result, fmt, ...)                                                    \
  do {                                                                                           \
    cudaError_t err = (cu_result);                                                               \
    if (err != cudaSuccess) {                                                                    \
      GXF_LOG_ERROR(fmt ", cuda_error: %s, error_str: %s", ##__VA_ARGS__, cudaGetErrorName(err), \
                    cudaGetErrorString(err));                                                    \
      throw std::runtime_error("Error occurred in CUDA runtime API call");                       \
    }                                                                                            \
  } while (0)

static constexpr const char* dlpack_capsule_name{"dltensor"};
static constexpr const char* used_dlpack_capsule_name{"used_dltensor"};

// Gets Python buffer format string from element type of the tensor
const char* PythonBufferFormatString(nvidia::gxf::PrimitiveType element_type) {
  switch (element_type) {
    case nvidia::gxf::PrimitiveType::kCustom:
      return "";
    case nvidia::gxf::PrimitiveType::kUnsigned8:
      return "B";
    case nvidia::gxf::PrimitiveType::kUnsigned16:
      return "H";
    case nvidia::gxf::PrimitiveType::kUnsigned32:
      return "I";
    case nvidia::gxf::PrimitiveType::kUnsigned64:
      return "Q";
    case nvidia::gxf::PrimitiveType::kInt8:
      return "b";
    case nvidia::gxf::PrimitiveType::kInt16:
      return "h";
    case nvidia::gxf::PrimitiveType::kInt32:
      return "i";
    case nvidia::gxf::PrimitiveType::kInt64:
      return "q";
    case nvidia::gxf::PrimitiveType::kFloat16:
      return "e";
    case nvidia::gxf::PrimitiveType::kFloat32:
      return "f";
    case nvidia::gxf::PrimitiveType::kFloat64:
      return "d";
    case nvidia::gxf::PrimitiveType::kComplex64:
      return "Zf";
    case nvidia::gxf::PrimitiveType::kComplex128:
      return "Zd";
  }
  return "";
}

/**
 * @brief Structure to hold the context of a DLManagedTensor.
 *
 * This structure is used to hold the context of a DLManagedTensor for array interface.
 */
struct ArrayInterfaceMemoryBuffer {
  pybind11::object obj_ref;         ///< Reference to the Python object that owns the memory buffer.
  std::vector<int64_t> dl_shape;    ///< Shape of the DLManagedTensor.
  std::vector<int64_t> dl_strides;  ///< Strides of the DLManagedTensor.
};

/**
 * @brief Set the array interface object of a Python object.
 *
 * This method sets `__array_interface__` or `__cuda_array_interface__` attribute of a Python
 * object.
 *
 * @param obj The Python object to set the array interface object.
 * @param ctx The context of the DLManagedTensor.
 */
void set_array_interface(const pybind11::object& obj,
                         std::shared_ptr<nvidia::gxf::DLManagedTensorContext> ctx);

using DLTensorType = nvidia::gxf::Tensor;

/**
 * @brief Provide `__dlpack__` method
 *
 * @param tensor The tensor to provide the `__dlpack__` method.
 * @param stream The client stream to use for the `__dlpack__` method.
 * @return The PyCapsule object.
 */
pybind11::capsule py_dlpack(DLTensorType* tensor, pybind11::object stream);

/**
 * @brief Provide `__dlpack_device__` method
 *
 * @param tensor The tensor to provide the `__dlpack_device__` method.
 * @return The tuple of device type and device id.
 */
pybind11::tuple py_dlpack_device(DLTensorType* tensor);

/**
 * @brief Create a new Tensor object from a pybind11::object
 *
 * The given pybind11::object must support the array interface protocol or dlpack protocol.
 *
 * @param obj A pybind11::object that can be converted to a Tensor
 * @return A new Tensor object
 */
pybind11::object as_tensor(const pybind11::object& obj);

// LazyDLManagedTensorDeleter was originally introduced in Holoscan SDK's Python API to address
// a potential deadlock due to Python GIL contention when running distributed applications with
// the multi-thread scheduler and GXF UCX extension. See details in: NVBUG 4293741.

/**
 * @brief A class facilitating lazy, asynchronous deletion of DLManagedTensor objects.
 *
 * This class allows DLManagedTensor objects to be enqueued for deferred deletion, which is carried
 * out in a distinct thread to evade the obstruction of the main execution thread.
 *
 * Instances of LazyDLManagedTensorDeleter are reference-counted. The thread responsible for
 * deletion is initiated upon the creation of the first instance and is ceased upon the destruction
 * of the last existing instance. The add() method can be employed to insert DLManagedTensor objects
 * into the deletion queue. The class destructor ensures the completion of all pending deletions
 * before finalizing.
 */
class LazyDLManagedTensorDeleter {
 public:
  /**
   * @brief Default constructor that initializes the LazyDLManagedTensorDeleter instance.
   *
   * Increment the reference count and start the deletion thread if it hasn't already started.
   * Register the pthread_atfork() and atexit() handlers if they aren't already registered.
   */
  LazyDLManagedTensorDeleter();

  /**
   * @brief Destructor that decrements the reference count and stops the deletion thread if the
   * count reaches zero.
   */
  ~LazyDLManagedTensorDeleter();

  /**
   * @brief Adds a DLManagedTensor pointer to the queue for deletion.
   * @param dl_managed_tensor_ptr The pointer to the DLManagedTensor to be deleted.
   */
  static void add(DLManagedTensor* dl_managed_tensor_ptr);

 private:
  /**
   * @brief The main function for the deletion thread, which waits for tensors to be available in
   * the queue and deletes them.
   */
  static void run();

  /**
   * @brief Decrements the reference count and stops the deletion thread if the count reaches zero.
   */
  static void release();

  /// Callback function for the atexit() function.
  static void on_exit();
  /// Callback function for the pthread_atfork() function's prepare handler.
  static void on_fork_prepare();
  /// Callback function for the pthread_atfork() function's parent handler.
  static void on_fork_parent();
  /// Callback function for the pthread_atfork() function's child handler.
  static void on_fork_child();

  ///< The queue of DLManagedTensor pointers to be deleted.
  static inline std::queue<DLManagedTensor*> s_dlmanaged_tensors_queue;
  ///< Mutex to protect the shared resources (queue, condition variable, etc.)
  static inline std::mutex s_mutex;
  ///< Condition variable to synchronize the deletion thread.
  static inline std::condition_variable s_cv;
  ///< A flag indicating whether the atfork handlers have been registered.
  static inline bool s_pthread_atfork_registered = false;
  ///< A flag indicating whether s_cv should not wait for the deletion thread so that fork() can
  ///< work.
  static inline bool s_cv_do_not_wait_thread = false;
  ///< The deletion thread.
  static inline std::thread s_thread;
  ///< The reference count of LazyDLManagedTensorDeleter instances.
  static inline std::atomic<int64_t> s_instance_count{0};
  ///< A flag indicating whether the deletion thread should stop.
  static inline bool s_stop = false;
  ///< A flag indicating whether the deletion thread is running.
  static inline bool s_is_running = false;
};

/**
 * @brief Class to wrap the deleter of a DLManagedTensor in Python.
 *
 * Compared to the C++ version (DLManagedMemoryBuffer), this class is used to acquire the GIL
 * before calling the deleter function.
 *
 */
class PyDLManagedMemoryBuffer {
 public:
  explicit PyDLManagedMemoryBuffer(DLManagedTensor* self);
  ~PyDLManagedMemoryBuffer();

 private:
  DLManagedTensor* self_ = nullptr;
};

PyDLManagedMemoryBuffer::PyDLManagedMemoryBuffer(DLManagedTensor* self) : self_(self) {}

PyDLManagedMemoryBuffer::~PyDLManagedMemoryBuffer() {
  // Add the DLManagedTensor pointer to the queue for asynchronous deletion.
  // Without this, the deleter function will be called immediately, which can cause deadlock
  // when the deleter function is called from another non-python thread with GXF runtime mutex
  // acquired (issue 4293741).
  LazyDLManagedTensorDeleter::add(self_);
}

class PyTensor : public nvidia::gxf::Tensor {
 public:
  /**
   * @brief Construct a new Tensor from an existing DLManagedTensorContext.
   *
   * @param ctx A shared pointer to the DLManagedTensorContext to be used in Tensor construction.
   */
  explicit PyTensor(std::shared_ptr<nvidia::gxf::DLManagedTensorContext>& ctx);

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensor pointer.
   *
   * @param ctx A pointer to the DLManagedTensor to be used in Tensor construction.
   */
  explicit PyTensor(DLManagedTensor* dl_managed_tensor_ptr);

  PyTensor() = default;

  PyTensor(Tensor&& other) { *this = std::move(other); }

  // static std::shared_ptr<PyTensor> from_dlpack(const pybind11::object& obj);
  // static pybind11::capsule dlpack(const pybind11::object& obj, pybind11::object stream);
  // static pybind11::tuple dlpack_device(const pybind11::object& obj);
};

PyTensor::PyTensor(std::shared_ptr<nvidia::gxf::DLManagedTensorContext>& ctx)
    : nvidia::gxf::Tensor(ctx) {}

PyTensor::PyTensor(DLManagedTensor* dl_managed_tensor_ptr) {
  dl_ctx_ = std::make_shared<nvidia::gxf::DLManagedTensorContext>();
  // Create PyDLManagedMemoryBuffer to hold the DLManagedTensor and acquire GIL before calling
  // the deleter function
  dl_ctx_->memory_ref = std::make_shared<PyDLManagedMemoryBuffer>(dl_managed_tensor_ptr);

  auto& dl_managed_tensor = dl_ctx_->tensor;
  dl_managed_tensor = *dl_managed_tensor_ptr;
  fromDLPack(dl_ctx_);
}

using ReturnTensorType = PyTensor;  // nvidia::gxf::Tensor;  // PyTensor

std::shared_ptr<ReturnTensorType> from_array_interface(const pybind11::object& obj, bool cuda) {
  auto memory_buf = std::make_shared<ArrayInterfaceMemoryBuffer>();
  memory_buf->obj_ref = obj;  // hold obj to prevent it from being garbage collected

  const char* interface_name = cuda ? "__cuda_array_interface__" : "__array_interface__";
  auto array_interface = obj.attr(interface_name).cast<pybind11::dict>();

  // Process mandatory entries
  memory_buf->dl_shape = array_interface["shape"].cast<std::vector<int64_t>>();
  auto& shape = memory_buf->dl_shape;
  auto typestr = array_interface["typestr"].cast<std::string>();
  if (!cuda) {
    if (!array_interface.contains("data")) {
      throw std::runtime_error(
          "Array interface data entry is missing (buffer interface) which is not supported ");
    }
    auto data_obj = array_interface["data"];
    if (data_obj.is_none()) {
      throw std::runtime_error(
          "Array interface data entry is None (buffer interface) which is not supported");
    }
    if (!pybind11::isinstance<pybind11::tuple>(data_obj)) {
      throw std::runtime_error(
          "Array interface data entry is not a tuple (buffer interface) which is not supported");
    }
  }
  auto data_array = array_interface["data"].cast<std::vector<int64_t>>();
  auto data_ptr = reinterpret_cast<void*>(data_array[0]);
  // bool data_readonly = data_array[1] > 0;
  // auto version = array_interface["version"].cast<int64_t>();
  auto maybe_dldatatype = nvidia::gxf::DLDataTypeFromTypeString(typestr);
  if (!maybe_dldatatype) {
    throw std::runtime_error("Unable to determine DLDataType from NumPy typestr");
  }
  auto maybe_device = nvidia::gxf::DLDeviceFromPointer(data_ptr);
  if (!maybe_device) { throw std::runtime_error("Unable to determine DLDevice from data pointer"); }
  DLTensor local_dl_tensor{
      .data = data_ptr,
      .device = maybe_device.value(),
      .ndim = static_cast<int32_t>(shape.size()),
      .dtype = maybe_dldatatype.value(),
      .shape = shape.data(),
      .strides = nullptr,
      .byte_offset = 0,
  };

  // Process 'optional' entries
  pybind11::object strides_obj = pybind11::none();
  if (array_interface.contains("strides")) { strides_obj = array_interface["strides"]; }
  auto& strides = memory_buf->dl_strides;
  if (strides_obj.is_none()) {
    nvidia::gxf::ComputeDLPackStrides(local_dl_tensor, strides, true);
  } else {
    strides = strides_obj.cast<std::vector<int64_t>>();
    // The array interface's stride is using bytes, not element size, so we need to divide it by
    // the element size.
    int64_t elem_size = local_dl_tensor.dtype.bits / 8;
    for (auto& stride : strides) { stride /= elem_size; }
  }
  local_dl_tensor.strides = strides.data();

  // We do not process 'descr', 'mask', and 'offset' entries

  // Process 'stream' entry
  if (cuda) {
    pybind11::object stream_obj = pybind11::none();
    if (array_interface.contains("stream")) { stream_obj = array_interface["stream"]; }

    int64_t stream_id = 1;  // legacy default stream
    cudaStream_t stream_ptr = nullptr;
    if (stream_obj.is_none()) {
      stream_id = -1;
    } else {
      stream_id = stream_obj.cast<int64_t>();
    }
    if (stream_id < -1) {
      throw std::runtime_error(
          "Invalid stream, valid stream should be None (no synchronization), 1 (legacy default "
          "stream), 2 (per-thread defaultstream), or a positive integer (stream pointer)");
    } else if (stream_id <= 2) {
      stream_ptr = nullptr;
    } else {
      stream_ptr = reinterpret_cast<cudaStream_t>(stream_id);
    }

    cudaStream_t curr_stream_ptr = nullptr;  // legacy stream

    if (stream_id >= 0 && curr_stream_ptr != stream_ptr) {
      cudaEvent_t curr_stream_event;
      cudaError_t cuda_status;

      cuda_status = cudaEventCreateWithFlags(&curr_stream_event, cudaEventDisableTiming);
      CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaEventCreateWithFlags");

      cuda_status = cudaEventRecord(curr_stream_event, stream_ptr);
      CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaEventRecord");
      // Make current stream (curr_stream_ptr) to wait until the given stream (stream_ptr)
      // is finished. This is a reverse of py_dlpack() method.
      cuda_status = cudaStreamWaitEvent(curr_stream_ptr, curr_stream_event, 0);
      CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaStreamWaitEvent");

      cuda_status = cudaEventDestroy(curr_stream_event);
      CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaEventDestroy");
    }
  }

  // Create DLManagedTensor object
  auto dl_managed_tensor_ctx = new nvidia::gxf::DLManagedTensorContext;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  dl_managed_tensor_ctx->memory_ref = memory_buf;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx =
        static_cast<nvidia::gxf::DLManagedTensorContext*>(self->manager_ctx);
    // Note: since 'memory_ref' is maintaining python object reference, we should acquire GIL in
    // case this function is called from another non-python thread, before releasing 'memory_ref'.
    pybind11::gil_scoped_acquire scope_guard;
    dl_managed_tensor_ctx->memory_ref.reset();
    delete dl_managed_tensor_ctx;
  };

  // Copy the DLTensor struct data
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor = local_dl_tensor;

  // Create PyTensor
  std::shared_ptr<ReturnTensorType> tensor = std::make_shared<ReturnTensorType>(&dl_managed_tensor);

  return tensor;
}

/**
 * @brief Convert an array to pybind11::tuple
 *
 * @tparam PT The type of the elements in the tuple.
 * @tparam T The type of the elements in the array.
 * @param vec The vector to convert.
 * @return The Python tuple object.
 */
template <typename PT, typename T>
pybind11::tuple array2pytuple(const T* arr, ssize_t length) {
  pybind11::tuple result(length);
  for (int index = 0; index < length; ++index) {
    const auto& value = arr[index];
    PyTuple_SET_ITEM(result.ptr(), index,
                     pybind11::reinterpret_steal<pybind11::object>(
                         pybind11::detail::make_caster<PT>::cast(
                             std::forward<PT>(value),
                             pybind11::return_value_policy::automatic_reference, nullptr))
                         .release()
                         .ptr());
  }
  return result;
}

void set_array_interface(const pybind11::object& obj,
                         std::shared_ptr<nvidia::gxf::DLManagedTensorContext> ctx) {
  DLTensor& dl_tensor = ctx->tensor.dl_tensor;

  if (dl_tensor.data) {
    // Prepare the array interface items

    // Main items
    auto maybe_type_str = nvidia::gxf::numpyTypestr(dl_tensor.dtype);
    if (!maybe_type_str) {
      throw std::runtime_error("Unable to determine NumPy dtype from DLPack tensor");
    }
    const char* type_str = maybe_type_str.value();
    pybind11::tuple shape = array2pytuple<pybind11::int_>(dl_tensor.shape, dl_tensor.ndim);
    pybind11::str typestr = pybind11::str(type_str);
    pybind11::tuple data = pybind11::make_tuple(
        pybind11::int_(reinterpret_cast<uint64_t>(dl_tensor.data)), pybind11::bool_(false));
    // Optional items
    pybind11::object strides = pybind11::none();
    if (dl_tensor.strides) {
      const int32_t strides_length = dl_tensor.ndim;
      pybind11::tuple strides_tuple(strides_length);
      // The array interface's stride is using bytes, not element size, so we need to multiply it by
      // the element size.
      auto& strides_arr = dl_tensor.strides;
      int64_t elem_size = dl_tensor.dtype.bits / 8;
      for (int index = 0; index < strides_length; ++index) {
        const auto& value = strides_arr[index];
        strides_tuple[index] = pybind11::int_(value * elem_size);
      }

      strides = strides_tuple;
    }
    pybind11::list descr;
    descr.append(pybind11::make_tuple("", typestr));

    // Depending on container's memory type, expose either array_interface or cuda_array_interface
    switch (dl_tensor.device.device_type) {
      case kDLCPU:
      case kDLCUDAHost: {
        // Reference: https://numpy.org/doc/stable/reference/arrays.interface.html
        obj.attr("__array_interface__") =
            pybind11::dict{"shape"_a = shape,     "typestr"_a = typestr,
                           "data"_a = data,       "version"_a = pybind11::int_(3),
                           "strides"_a = strides, "descr"_a = descr};
      } break;
      case kDLCUDA:
      case kDLCUDAManaged: {
        // Reference: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        // TODO(gbae): Add support for stream instead of always using the default stream
        obj.attr("__cuda_array_interface__") = pybind11::dict{
            "shape"_a = shape,
            "typestr"_a = typestr,
            "data"_a = data,
            "version"_a = pybind11::int_(3),
            "strides"_a = strides,
            "descr"_a = descr,
            "mask"_a = pybind11::none(),
            "stream"_a = 1  // 1: The legacy default stream
        };
      } break;
      default:
        break;
    }
  } else {
    switch (dl_tensor.device.device_type) {
      case kDLCPU:
      case kDLCUDAHost: {
        if (pybind11::hasattr(obj, "__array_interface__")) {
          pybind11::delattr(obj, "__array_interface__");
        }
      } break;
      case kDLCUDA:
      case kDLCUDAManaged: {
        if (pybind11::hasattr(obj, "__cuda_array_interface__")) {
          pybind11::delattr(obj, "__cuda_array_interface__");
        }
      } break;
      default:
        break;
    }
  }
}

pybind11::capsule py_dlpack(DLTensorType* tensor, pybind11::object stream) {
  // TOIMPROVE: need to get current stream pointer and call with the stream
  cudaStream_t curr_stream_ptr = nullptr;  // legacy stream

  int64_t stream_id = 1;  // legacy default stream
  cudaStream_t stream_ptr = nullptr;

  if (stream.is_none()) {
    stream = pybind11::int_(1);  // legacy default stream
  } else if (pybind11::isinstance<pybind11::int_>(stream)) {
    stream_id = stream.cast<int64_t>();
    if (stream_id < -1) {
      throw std::runtime_error(
          "Invalid stream, valid stream should be -1 (non-blocking), 1 (legacy default stream), 2 "
          "(per-thread default stream), or a positive integer (stream pointer)");
    } else if (stream_id <= 2) {
      // Allow the stream id 0 as a special case for the default stream.
      // This is to support the legacy behavior.
      stream_ptr = nullptr;
    } else {
      stream_ptr = reinterpret_cast<cudaStream_t>(stream_id);
    }
  } else {
    throw std::runtime_error("Invalid stream type: should be int type but given '"s +
                             std::string(pybind11::str(stream)) + "'"s);
  }

  // Wait for the current stream to finish before the provided stream starts consuming the memory.
  if (stream_id >= 0 && curr_stream_ptr != stream_ptr) {
    cudaEvent_t curr_stream_event;
    cudaError_t cuda_status;

    cuda_status = cudaEventCreateWithFlags(&curr_stream_event, cudaEventDisableTiming);
    CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaEventCreateWithFlags");

    cuda_status = cudaEventRecord(curr_stream_event, curr_stream_ptr);
    CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaEventRecord");

    cuda_status = cudaStreamWaitEvent(stream_ptr, curr_stream_event, 0);
    CHECK_CUDA_THROW_ERROR(cuda_status, "Failure during call to cudaStreamWaitEvent");

    cuda_status = cudaEventDestroy(curr_stream_event);
  }

  auto maybe_dl_managed_tensor = tensor->toDLPack();
  if (!maybe_dl_managed_tensor) {
    throw std::runtime_error("Failed to export DLManagedTensor* via Tensor::toDLPack");
  }
  DLManagedTensor* dl_managed_tensor = maybe_dl_managed_tensor.value();

  // Create a new capsule to hold the DLPack tensor. The destructor of the capsule will call
  // `DLManagedTensor::deleter` to free the memory. The destructor will be called when the capsule
  // goes out of scope or when the capsule is destroyed.
  pybind11::capsule dlpack_capsule(dl_managed_tensor, dlpack_capsule_name, [](PyObject* ptr) {
    // Should call `PyCapsule_IsValid` to check if the capsule is valid before calling
    // `PyCapsule_GetPointer`. Otherwise, it will raise a hard-to-debug exception.
    // (such as `SystemError: <class 'xxx'> returned a result with an error set`)
    if (PyCapsule_IsValid(ptr, dlpack_capsule_name)) {
      // The destructor will be called when the capsule is deleted.
      // We need to call the deleter function to free the memory.
      DLManagedTensor* dl_managed_tensor =
          static_cast<DLManagedTensor*>(PyCapsule_GetPointer(ptr, dlpack_capsule_name));
      // Call deleter function to free the memory only if the capsule name is "dltensor".
      if (dl_managed_tensor != nullptr) { dl_managed_tensor->deleter(dl_managed_tensor); }
    }
  });

  return dlpack_capsule;
}

pybind11::tuple py_dlpack_device(DLTensorType* tensor) {
  auto maybe_dl_managed_tensor = tensor->toDLManagedTensorContext();
  if (!maybe_dl_managed_tensor) {
    throw std::runtime_error("Failed to get DLManagedTensorContext from the tensor");
  }
  auto& dl_tensor = (maybe_dl_managed_tensor.value())->tensor.dl_tensor;
  auto& device = dl_tensor.device;
  auto& device_type = device.device_type;
  auto& device_id = device.device_id;
  return pybind11::make_tuple(pybind11::int_(static_cast<int>(device_type)),
                              pybind11::int_(device_id));
}

std::shared_ptr<PyTensor> from_dlpack(const pybind11::object& obj) {
  // Pybind11 doesn't have a way to get/set a pointer with a name so we have to use the C API
  // for efficiency.
  // auto dlpack_capsule =
  // pybind11::reinterpret_borrow<pybind11::capsule>(obj.attr("__dlpack__")());
  auto dlpack_device_func = obj.attr("__dlpack_device__");

  // We don't handle backward compatibility with older versions of DLPack
  if (dlpack_device_func.is_none()) { throw std::runtime_error("DLPack device is not set"); }

  auto dlpack_device = pybind11::cast<pybind11::tuple>(dlpack_device_func());
  // https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv48DLDevice
  DLDeviceType device_type = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
  int32_t device_id = dlpack_device[1].cast<int32_t>();

  DLDevice device = {device_type, device_id};

  auto dlpack_func = obj.attr("__dlpack__");
  pybind11::capsule dlpack_capsule;

  // TOIMPROVE: need to get current stream pointer and call with the stream
  // https://github.com/dmlc/dlpack/issues/57 this thread was good to understand the differences
  // between __cuda_array_interface__ and __dlpack__ on life cycle/stream handling.
  // In DLPack, the client of the memory notify to the producer that the client will use the
  // client stream (`stream_ptr`) to consume the memory. It's the producer's responsibility to
  // make sure that the client stream wait for the producer stream to finish producing the memory.
  // The producer stream is the stream that the producer used to produce the memory. The producer
  // can then use this information to decide whether to use the same stream to produce the memory
  // or to use a different stream.
  // In __cuda_array_interface__, both producer and consumer are responsible for managing the
  // streams. The producer can use the `stream` field to specify the stream that the producer used
  // to produce the memory. The consumer can use the `stream` field to synchronize with the
  // producer stream. (please see
  // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html#synchronization)
  switch (device_type) {
    case kDLCUDA:
    case kDLCUDAManaged: {
      pybind11::int_ stream_ptr(1);  // legacy stream
      dlpack_capsule = pybind11::reinterpret_borrow<pybind11::capsule>(
          dlpack_func(pybind11::arg("stream") = stream_ptr));
      break;
    }
    case kDLCPU:
    case kDLCUDAHost: {
      dlpack_capsule = pybind11::reinterpret_borrow<pybind11::capsule>(dlpack_func());
      break;
    }
    default:
      throw std::runtime_error("Unsupported device type");
  }

  // Note: we should keep the reference to the capsule object (`dlpack_obj`) while working with
  // PyObject* pointer. Otherwise, the capsule can be deleted and the pointers will be invalid.
  pybind11::object dlpack_obj = dlpack_func();

  PyObject* dlpack_capsule_ptr = dlpack_obj.ptr();

  if (!PyCapsule_IsValid(dlpack_capsule_ptr, dlpack_capsule_name)) {
    std::string capsule_name{PyCapsule_GetName(dlpack_capsule_ptr)};
    throw std::runtime_error(
        "Received an invalid DLPack capsule (" + capsule_name + "). You might have already consumed "
        "the DLPack capsule.");
  }

  DLManagedTensor* dl_managed_tensor =
      static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(dlpack_capsule_ptr, dlpack_capsule_name));

  // Set device
  dl_managed_tensor->dl_tensor.device = device;

  // Create PyTensor
  std::shared_ptr<PyTensor> tensor = std::make_shared<PyTensor>(dl_managed_tensor);

  // Set the capsule name to 'used_dltensor' so that it will not be consumed again.
  PyCapsule_SetName(dlpack_capsule_ptr, used_dlpack_capsule_name);

  // Steal the ownership of the capsule so that it will not be destroyed when the capsule object
  // goes out of scope.
  PyCapsule_SetDestructor(dlpack_capsule_ptr, nullptr);

  return tensor;
}

pybind11::object set_array_interface_on_tensor(std::shared_ptr<ReturnTensorType> tensor) {
  pybind11::object py_tensor = pybind11::cast(tensor);

  // Set array interface attributes
  auto maybe_dl_ctx = tensor->toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    throw std::runtime_error("Unable to export Tensor to DLManagedTensorContext");
  }
  set_array_interface(py_tensor, maybe_dl_ctx.value());
  return py_tensor;
}

pybind11::object as_tensor(const pybind11::object& obj) {
  // This method could have been used as a constructor for the PyTensor class, but it was not
  // possible to get the pybind11::object to be passed to the constructor. Instead, this method is
  // used to create a pybind11::object from PyTensor object and set array interface on it.
  //
  //    // Note: this does not work, as the pybind11::object is not passed to the constructor
  //    .def(pybind11::init(&PyTensor::py_create), doc::Tensor::doc_Tensor);
  //
  //       include/pybind11/detail/init.h:86:19: error: static assertion failed: pybind11::init():
  //       init function must return a compatible pointer, holder, or value
  //       86 |     static_assert(!std::is_same<Class, Class>::value /* always false */,
  //
  //    // See https://github.com/pybind/pybind11/issues/2984 for more details
  std::shared_ptr<ReturnTensorType> tensor;

  if (pybind11::hasattr(obj, "__cuda_array_interface__")) {
    tensor = from_array_interface(obj, true);
  } else if (pybind11::hasattr(obj, "__array_interface__")) {
    tensor = from_array_interface(obj, false);
  } else if (pybind11::hasattr(obj, "__dlpack__") && pybind11::hasattr(obj, "__dlpack_device__")) {
    tensor = from_dlpack(obj);
  } else {
    throw std::runtime_error("Unsupported Python object type");
  }
  return set_array_interface_on_tensor(tensor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// LazyDLManagedTensorDeleter
////////////////////////////////////////////////////////////////////////////////////////////////////

LazyDLManagedTensorDeleter::LazyDLManagedTensorDeleter() {

  GXF_LOG_DEBUG("LazyDLManagedTensorDeleter() constructor");
  // Use std::memory_order_relaxed because there are no other memory operations that need to be
  // synchronized with the fetch_add operation.
  if (s_instance_count.fetch_add(1, std::memory_order_relaxed) == 0) {
    // Wait until both s_stop and s_is_running are false (busy-waiting).
    // s_stop being true indicates that the previous deleter thread is still in the process
    // of deleting the object.
    while (true) {
      {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_stop && !s_is_running) { break; }
      }
      // Yield to other threads
      std::this_thread::yield();
    }

    std::lock_guard<std::mutex> lock(s_mutex);
    // Register pthread_atfork() and std::atexit() handlers (registered only once)
    //
    // Note: NVBUG 4318040
    // When fork() is called in a multi-threaded program, the child process will only have
    // the thread that called fork().
    // Other threads from the parent process won't be running in the child.
    // This can lead to deadlocks if a condition variable or mutex was being waited upon by another
    // thread at the time of the fork.
    // To avoid this, we register pthread_atfork() handlers to acquire all necessary locks in
    // the pre-fork handler and release them in both post-fork handlers, ensuring no mutex or
    // condition variable remains locked in the child.
    if (!s_pthread_atfork_registered) {
      pthread_atfork(on_fork_prepare, on_fork_parent, on_fork_child);
      s_pthread_atfork_registered = true;
      // Register on_exit() to be called when the application exits.
      // Note that the child process will not call on_exit() when fork() is called and exit() is
      // called in the child process.
      std::atexit(on_exit);
    }

    s_is_running = true;
    s_thread = std::thread(run);
    // Detach the thread so that it can be stopped when the application exits
    //
    // Note: NVBUG 4318040
    // According to the C++ Core Guidelines in CP.24 and CP.26
    // (https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines), std::detach() is generally
    // discouraged.
    // In C++ 20, std::jthread will be introduced to replace std::thread, and std::thread::detach()
    // will be deprecated.
    // However, std::jthread is not available in C++ 17 and we need to use std::thread::detach()
    // for now, with a synchronization mechanism to wait for the thread to finish itself,
    // instead of introducing a new dependency like https://github.com/martinmoene/jthread-lite.
    s_thread.detach();
  }
}

LazyDLManagedTensorDeleter::~LazyDLManagedTensorDeleter() {
  release();
}

void LazyDLManagedTensorDeleter::add(DLManagedTensor* dl_managed_tensor_ptr) {
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_dlmanaged_tensors_queue.push(dl_managed_tensor_ptr);
  }
  s_cv.notify_all();
}

void LazyDLManagedTensorDeleter::run() {
  GXF_LOG_DEBUG("LazyDLManagedTensorDeleter run called");
  while (true) {
    std::unique_lock<std::mutex> lock(s_mutex);

    s_cv.wait(lock, [] {
      return s_stop || !s_dlmanaged_tensors_queue.empty() || s_cv_do_not_wait_thread;
    });

    // Check if the thread should stop. If queue is not empty, process the queue.
    if (s_stop && s_dlmanaged_tensors_queue.empty()) { break; }

    // Check if the condition variable should not wait for the thread so that fork() can be called
    // without deadlock.
    if (s_cv_do_not_wait_thread) { continue; }

    // move queue onto the local stack before releasing the lock
    std::queue<DLManagedTensor*> local_queue;
    local_queue.swap(s_dlmanaged_tensors_queue);

    lock.unlock();
    // Call the deleter function for each pointer in the queue
    while (!local_queue.empty()) {
      auto dl_managed_tensor_ptr = local_queue.front();
      // Note: the deleter function can be nullptr (e.g. when the tensor is created from
      // __cuda_array_interface__ protocol)
      if (dl_managed_tensor_ptr && dl_managed_tensor_ptr->deleter != nullptr) {
        GXF_LOG_INFO("Calling deleter from LazyDLManagedTensorDeleter::run");
        // Call the deleter function with GIL acquired
        pybind11::gil_scoped_acquire scope_guard;
        dl_managed_tensor_ptr->deleter(dl_managed_tensor_ptr);
      }
      local_queue.pop();
    }
  }

  // Set the flag to indicate that the thread has stopped
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_is_running = false;
  }
  GXF_LOG_DEBUG("LazyDLManagedTensorDeleter thread finished");
}

void LazyDLManagedTensorDeleter::release() {
  // Use std::memory_order_relaxed because there are no other memory operations that need to be
  // synchronized with the fetch_sub operation.
  if (s_instance_count.fetch_sub(1, std::memory_order_relaxed) == 1) {
    {
      std::lock_guard<std::mutex> lock(s_mutex);
      s_stop = true;
    }
    s_cv.notify_all();
    GXF_LOG_DEBUG("Waiting for LazyDLManagedTensorDeleter thread to stop");
    // Wait until the thread has stopped
    while (true) {
      {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_is_running) { break; }
      }
      // Yield to other threads
      std::this_thread::yield();
    }
    GXF_LOG_DEBUG("LazyDLManagedTensorDeleter thread stopped");
    {
      std::lock_guard<std::mutex> lock(s_mutex);
      s_stop = false;
    }

  }
}

void LazyDLManagedTensorDeleter::on_exit() {
  GXF_LOG_DEBUG("LazyDLManagedTensorDeleter::on_exit() called");
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_stop = true;
  }
  s_cv.notify_all();
}

void LazyDLManagedTensorDeleter::on_fork_prepare() {
  s_mutex.lock();
  LazyDLManagedTensorDeleter::s_cv_do_not_wait_thread = true;
  s_cv.notify_all();
}

void LazyDLManagedTensorDeleter::on_fork_parent() {
  s_mutex.unlock();
  LazyDLManagedTensorDeleter::s_cv_do_not_wait_thread = false;
}

void LazyDLManagedTensorDeleter::on_fork_child() {
  s_mutex.unlock();
  LazyDLManagedTensorDeleter::s_cv_do_not_wait_thread = false;
}

class PyLazyDLManagedTensorDeleter {
 public:
  PyLazyDLManagedTensorDeleter() = default;

  ~PyLazyDLManagedTensorDeleter() = default;

 private:
  LazyDLManagedTensorDeleter deleter{};
};

}  // namespace

PYBIND11_MODULE(tensor_pybind, m) {
  m.doc() = R"pbdoc(
        Python bridge for accessing tensor buffer
        -----------------------

        .. currentmodule:: pygxf

    )pbdoc";

  // Define a pybind buffer parser that converts tensors to numpy array automatically on return
  pybind11::class_<nvidia::gxf::Handle<nvidia::gxf::Tensor>>(m, "TensorHandle",
                                                             pybind11::buffer_protocol())
      .def_buffer([](nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor) -> pybind11::buffer_info {
        std::vector<int> dimensions;
        std::vector<int> strides;
        const char* format = PythonBufferFormatString(tensor->element_type());
        nvidia::gxf::Shape tensor_shape = tensor->shape();
        for (size_t i = 0; i < tensor_shape.rank(); i++) {
          dimensions.push_back(tensor_shape.dimension(i));
          strides.push_back(tensor->stride(i));
        }
        return pybind11::buffer_info(tensor->pointer(),           /* Pointer to buffer */
                                     tensor->bytes_per_element(), /* Size of one scalar */
                                     format, /* Python struct-style format descriptor */
                                     tensor_shape.rank(), /* Number of dimensions */
                                     dimensions,          /* Buffer dimensions */
                                     strides              /* Stride dimensions */
        );
      });

  m.def("get_from_entity_context", [](uint64_t context_idx, gxf_uid_t eid, const char* tensor_field) {
    const gxf_context_t context = reinterpret_cast<gxf_context_t>(context_idx);
    auto entity = nvidia::gxf::Entity::Shared(context, eid);
    auto maybe_tensor = entity.value().get<nvidia::gxf::Tensor>(tensor_field);
    if (!maybe_tensor) { throw pybind11::value_error("Field with matching name does not exist"); }
    return pybind11::array(pybind11::cast(maybe_tensor.value()));
  });

  pybind11::class_<nvidia::gxf::Tensor, std::shared_ptr<nvidia::gxf::Tensor>>(
      m, "Tensor", pybind11::dynamic_attr(), pybind11::buffer_protocol())
      .def_buffer([](nvidia::gxf::Tensor& tensor) -> pybind11::buffer_info {
        std::vector<int> dimensions;
        std::vector<int> strides;
        const char* format = PythonBufferFormatString(tensor.element_type());
        nvidia::gxf::Shape tensor_shape = tensor.shape();
        for (size_t i = 0; i < tensor_shape.rank(); i++) {
          dimensions.push_back(tensor_shape.dimension(i));
          strides.push_back(tensor.stride(i));
        }
        return pybind11::buffer_info(tensor.pointer(),           /* Pointer to buffer */
                                     tensor.bytes_per_element(), /* Size of one scalar */
                                     format, /* Python struct-style format descriptor */
                                     tensor_shape.rank(), /* Number of dimensions */
                                     dimensions,          /* Buffer dimensions */
                                     strides              /* Stride dimensions */
        );
      })
      .def(pybind11::init([]() { return nvidia::gxf::Tensor(); }),
           pybind11::return_value_policy::reference)
      .def(pybind11::init([](pybind11::array array) {
             static bool warned = false;
             if (!warned) {
               GXF_LOG_WARNING(
                 "Passing a NumPy array to the Tensor constructor is deprecated. Please use "
                 "`Tensor.as_tensor` instead for zero-copy initialization of a Tensor from an "
                 "existing array-like object.");
               warned = true;
             }
             auto t = nvidia::gxf::Tensor();
             std::array<int32_t, nvidia::gxf::Shape::kMaxRank> shape;
             auto array_shape = array.shape();
             for (auto i = 0; i < array.ndim(); i++) { shape[i] = *(array_shape + i); }
             std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides;
             auto array_strides = array.strides();
             for (auto i = 0; i < array.ndim(); i++) { strides[i] = *(array_strides + i); }
             nvidia::gxf::PrimitiveType element_type;
             if (pybind11::str(array.dtype()).equal(pybind11::str("int8")))
               element_type = nvidia::gxf::PrimitiveType::kInt8;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("uint8")))
               element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("int16")))
               element_type = nvidia::gxf::PrimitiveType::kInt16;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("uint16")))
               element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("int32")))
               element_type = nvidia::gxf::PrimitiveType::kInt32;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("uint32")))
               element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("int64")))
               element_type = nvidia::gxf::PrimitiveType::kInt64;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("uint64")))
               element_type = nvidia::gxf::PrimitiveType::kUnsigned64;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("float16")))
               element_type = nvidia::gxf::PrimitiveType::kFloat16;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("float32")))
               element_type = nvidia::gxf::PrimitiveType::kFloat32;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("float64")))
               element_type = nvidia::gxf::PrimitiveType::kFloat64;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("complex64")))
               element_type = nvidia::gxf::PrimitiveType::kComplex64;
             else if (pybind11::str(array.dtype()).equal(pybind11::str("complex128")))
               element_type = nvidia::gxf::PrimitiveType::kComplex128;
             else
               element_type = nvidia::gxf::PrimitiveType::kCustom;

             nvidia::gxf::MemoryBuffer::release_function_t release_func = [](void* pointer) {
               GXF_LOG_DEBUG("Tensor object deleted. No memory released");
               return nvidia::gxf::Success;
             };
             t.wrapMemory(nvidia::gxf::Shape(shape, static_cast<uint32_t>(array.ndim())),
                          element_type, static_cast<uint64_t>(array.itemsize()), strides,
                          nvidia::gxf::MemoryStorageType::kHost,
                          static_cast<void*>(array.request().ptr), release_func);
             return t;
           }),
           pybind11::return_value_policy::reference)
      .def(pybind11::init([](pybind11::object obj) {
            if (pybind11::hasattr(obj, "__cuda_array_interface__")
                || pybind11::hasattr(obj, "__array_interface__")
                || pybind11::hasattr(obj, "__dlpack__")) {
              throw pybind11::value_error(
                "Please use `Tensor.as_tensor(array_like)` to create a Tensor from an array-like "
                "object supporting the DLPack protocol, NumPy's array interface protocol or the "
                "CUDA array interface protocol.");
            }
            throw pybind11::value_error(
              "The `Tensor()` constructor can only create a tensor from a NumPy array. To "
              "create a Tensor from other array-like objects on host or device that support "
              "the DLPack protocol, NumPy's array interface protocol or the CUDA array interface "
              "protocol, use `Tensor.as_tensor()` instead.");
            return nvidia::gxf::Tensor();
          }),
          pybind11::return_value_policy::reference)
      .def("get_tensor_info",
           [](nvidia::gxf::Tensor& t) {
             auto rank = t.rank();
             std::vector<int32_t> dims;
             std::vector<int32_t> strides;
             void* buffer_ptr = static_cast<void*>(t.pointer());
             std::string descriptor = PythonBufferFormatString(t.element_type());
             for (uint i = 0; i < rank; i++) {
               strides.push_back(t.stride(i));
               dims.push_back(t.shape().dimension(i));
             }
             //  Pybind does not recognize Zf / Zd format strings
             if (descriptor == "Zf") descriptor = "complex64";
             if (descriptor == "Zd") descriptor = "complex128";
             return pybind11::make_tuple(pybind11::cast(buffer_ptr), t.size(),
                                         pybind11::dtype(descriptor), rank, dims, strides);
           })
      .def("shape", &nvidia::gxf::Tensor::shape)
      .def("element_type", &nvidia::gxf::Tensor::element_type)
      .def("storage_type", &nvidia::gxf::Tensor::storage_type)
      .def("reshape",
           [](nvidia::gxf::Tensor& t, nvidia::gxf::TensorDescription& td,
              nvidia::gxf::Allocator* allocator) {
             auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                 allocator->context(), allocator->cid());
             if (!allocator_handle) {
               throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
             }
             nvidia::gxf::Expected<void> result;
             if (sizeof(td.strides) == 0) {
               result = t.reshapeCustom(td.shape, td.element_type, td.bytes_per_element,
                                        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                        td.storage_type, allocator_handle.value());
             } else {
               result = t.reshapeCustom(td.shape, td.element_type, td.bytes_per_element, td.strides,
                                        td.storage_type, allocator_handle.value());
             }
             if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
             return;
           })
      .def("reshape_custom",
           [](nvidia::gxf::Tensor& t, const nvidia::gxf::Shape& shape,
              nvidia::gxf::PrimitiveType element_type, uint64_t bytes_per_element,
              std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides,
              nvidia::gxf::MemoryStorageType storage_type, nvidia::gxf::Allocator* allocator) {
             auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                 allocator->context(), allocator->cid());
             if (!allocator_handle) {
               throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
             }
             nvidia::gxf::Expected<void> result;
             if (strides.size() == 0) {
               result = t.reshapeCustom(shape, element_type, bytes_per_element,
                                        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                        storage_type, allocator_handle.value());
             } else {
               result = t.reshapeCustom(shape, element_type, bytes_per_element, strides,
                                        storage_type, allocator_handle.value());
             }
             if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
             return;
           })
      .def(
          "get_tensor_description",
          [](nvidia::gxf::Tensor& t) {
            nvidia::gxf::TensorDescription td;
            td.shape = t.shape();
            td.element_type = t.element_type();
            td.bytes_per_element = t.bytes_per_element();
            td.storage_type = t.storage_type();
            return td;
          },
          pybind11::return_value_policy::reference)
      .def(
          "add_np_array_as_tensor_to_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr, pybind11::array array,
             nvidia::gxf::Allocator * allocator) {
            // add a gxf::Tensor to the entity
            // GXF_LOG_INFO("This function will be deprecated soon. Please create a tensor from"
            // " numpy array using t = T(np.array([1, 2, 3])) and then add it to the message using"
            // " Tensor.add_to_entity(msg, t)");
            auto result = e.add<nvidia::gxf::Tensor>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            auto t = result.value().get();

            // calculate relevant info
            std::array<int32_t, nvidia::gxf::Shape::kMaxRank> shape;
            auto array_shape = array.shape();
            for (auto i = 0; i < array.ndim(); i++) { shape[i] = *(array_shape + i); }
            std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides;
            auto array_strides = array.strides();
            for (auto i = 0; i < array.ndim(); i++) { strides[i] = *(array_strides + i); }
            nvidia::gxf::PrimitiveType element_type;
            if (pybind11::str(array.dtype()).equal(pybind11::str("int8")))
              element_type = nvidia::gxf::PrimitiveType::kInt8;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint8")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("int16")))
              element_type = nvidia::gxf::PrimitiveType::kInt16;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint16")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("int32")))
              element_type = nvidia::gxf::PrimitiveType::kInt32;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint32")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("int64")))
              element_type = nvidia::gxf::PrimitiveType::kInt64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("uint64")))
              element_type = nvidia::gxf::PrimitiveType::kUnsigned64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("float16")))
              element_type = nvidia::gxf::PrimitiveType::kFloat16;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("float32")))
              element_type = nvidia::gxf::PrimitiveType::kFloat32;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("float64")))
              element_type = nvidia::gxf::PrimitiveType::kFloat64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("complex64")))
              element_type = nvidia::gxf::PrimitiveType::kComplex64;
            else if (pybind11::str(array.dtype()).equal(pybind11::str("complex128")))
              element_type = nvidia::gxf::PrimitiveType::kComplex128;
            else
              element_type = nvidia::gxf::PrimitiveType::kCustom;

            // create a tensor description
            nvidia::gxf::TensorDescription td{
                name,
                nvidia::gxf::MemoryStorageType::kHost,
                nvidia::gxf::Shape(shape, static_cast<uint32_t>(array.ndim())),
                element_type,
                static_cast<uint64_t>(array.itemsize()),
                strides};

            // reshape the tensor
            auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                allocator->context(), allocator->cid());
            if (!allocator_handle) {
              throw pybind11::value_error(GxfResultStr(allocator_handle.error()));
            }

            nvidia::gxf::Expected<void> reshape_result;
            if (sizeof(td.strides) == 0) {
              reshape_result = t->reshapeCustom(td.shape, td.element_type, td.bytes_per_element,
                                                nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                                td.storage_type, allocator_handle.value());
            } else {
              reshape_result =
                  t->reshapeCustom(td.shape, td.element_type, td.bytes_per_element, td.strides,
                                   td.storage_type, allocator_handle.value());
            }
            if (!reshape_result) {
              throw pybind11::value_error(GxfResultStr(reshape_result.error()));
            }

            // copy the data from numpy array to this tensor
            std::memcpy(t->pointer(), array.data(), array.nbytes());
            return t;
          },
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto result = e.add<nvidia::gxf::Tensor>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            return result.value().get();
          },
          pybind11::return_value_policy::reference)
      .def_static(
          "add_to_entity",
          [](nvidia::gxf::Entity& e, nvidia::gxf::Tensor& t, const char* name = nullptr) {
            auto result = e.add<nvidia::gxf::Tensor>(name);
            if (!result) { throw pybind11::value_error(GxfResultStr(result.error())); }
            auto new_tensor = result.value().get();
            *new_tensor = std::move(t);
            return;
          },
          "Add a tensor to entity", pybind11::arg("message"), pybind11::arg("tensor"),
          pybind11::arg("name") = "", pybind11::return_value_policy::reference)
      .def_static(
          "get_from_entity",
          [](nvidia::gxf::Entity& e, const char* name = nullptr) {
            auto maybe_tensor = e.get<nvidia::gxf::Tensor>(name);
            if (!maybe_tensor) {
              GXF_LOG_ERROR("Error getting tensor called %s", name);
              throw pybind11::value_error("error getting tensor");
            }
            // return as PyTensor also supporting __array_interface__ or __cuda_array_interface__
            auto maybe_dl_ctx = maybe_tensor.value()->toDLManagedTensorContext();
            if (!maybe_dl_ctx) {
              GXF_LOG_ERROR("Error getting tensor called %s", name);
              throw pybind11::value_error("error getting DLManagedTensorContext");
            }
            auto tensor = std::make_shared<PyTensor>(maybe_dl_ctx.value());
            pybind11::object py_tensor_obj = pybind11::cast(tensor);
            set_array_interface(py_tensor_obj, maybe_dl_ctx.value());
            return py_tensor_obj;
          },
          pybind11::arg("entity") = nullptr, pybind11::arg("name") = nullptr,
          pybind11::return_value_policy::reference)
      .def(
          "find_all_from_entity",
          [](nvidia::gxf::Entity& e) {
            nvidia::FixedVector<nvidia::gxf::Handle<nvidia::gxf::Tensor>, kMaxComponents>
                components;
            auto maybe_tensors = e.findAll<nvidia::gxf::Tensor>();
            if (!maybe_tensors) { throw pybind11::value_error("error getting tensors"); }
            components = maybe_tensors.value();
            std::vector<pybind11::object> result;
            for (uint i = 0; i < components.size(); i++) {
              auto maybe_tensor = components.at(i);
              if (!maybe_tensor) { throw std::runtime_error("Error getting tensor"); }

              // return as PyTensor also supporting __array_interface__ or __cuda_array_interface__
              auto maybe_dl_ctx = maybe_tensor.value()->toDLManagedTensorContext();
              if (!maybe_dl_ctx) {
                GXF_LOG_ERROR("Error getting tensor");
                throw pybind11::value_error("error getting DLManagedTensorContext");
              }
              auto tensor = std::make_shared<PyTensor>(maybe_dl_ctx.value());
              pybind11::object py_tensor_obj = pybind11::cast(tensor);
              set_array_interface(py_tensor_obj, maybe_dl_ctx.value());
              result.push_back(py_tensor_obj);
            }
            return result;
          })
      // For creation via the DLPack protocol
      .def_static("from_dlpack",
          [](const pybind11::object& obj) {
            auto tensor = from_dlpack(obj);
            return set_array_interface_on_tensor(tensor);
          })
      // For creation via DLPack or one of the array interface protocols or the DLPack protocol
      .def_static("as_tensor", [](const pybind11::object& obj) { return as_tensor(obj); })
      // DLPack protocol is implemented by __dlpack__ and __dlpack_device__
      .def(
          "__dlpack__",
          [](const pybind11::object& obj, pybind11::object stream = pybind11::none()) {
            auto tensor = pybind11::cast<std::shared_ptr<PyTensor>>(obj);
            if (!tensor) { throw std::runtime_error("Failed to cast to Tensor"); }
            // Do not copy 'obj' or a shared pointer here in the lambda expression's initializer,
            // otherwise the refcount of it will be increased by 1 and prevent the object from being
            // destructed. Use a raw pointer here instead.
            return py_dlpack(tensor.get(), stream);
          },
          pybind11::arg("stream") = pybind11::none())
      .def("__dlpack_device__", [](const pybind11::object& obj) {
        auto tensor = pybind11::cast<std::shared_ptr<PyTensor>>(obj);
        if (!tensor) { throw std::runtime_error("Failed to cast to Tensor"); }
        // Do not copy 'obj' or a shared pointer here in the lambda expression's initializer,
        // otherwise the refcount of it will be increased by 1 and prevent the object from being
        // destructed. Use a raw pointer here instead.
        return py_dlpack_device(tensor.get());
      });

  pybind11::class_<PyTensor, nvidia::gxf::Tensor, std::shared_ptr<PyTensor>>(m, "PyTensor");

  pybind11::enum_<nvidia::gxf::PrimitiveType>(m, "PrimitiveType")
      .value("kCustom", nvidia::gxf::PrimitiveType::kCustom)
      .value("kInt8", nvidia::gxf::PrimitiveType::kInt8)
      .value("kUnsigned8", nvidia::gxf::PrimitiveType::kUnsigned8)
      .value("kInt16", nvidia::gxf::PrimitiveType::kInt16)
      .value("kUnsigned16", nvidia::gxf::PrimitiveType::kUnsigned16)
      .value("kInt32", nvidia::gxf::PrimitiveType::kInt32)
      .value("kUnsigned32", nvidia::gxf::PrimitiveType::kUnsigned32)
      .value("kInt64", nvidia::gxf::PrimitiveType::kInt64)
      .value("kUnsigned64", nvidia::gxf::PrimitiveType::kUnsigned64)
      .value("kFloat16", nvidia::gxf::PrimitiveType::kFloat16)
      .value("kFloat32", nvidia::gxf::PrimitiveType::kFloat32)
      .value("kFloat64", nvidia::gxf::PrimitiveType::kFloat64)
      .value("kComplex64", nvidia::gxf::PrimitiveType::kComplex64)
      .value("kComplex128", nvidia::gxf::PrimitiveType::kComplex128);

  pybind11::enum_<nvidia::gxf::MemoryStorageType>(m, "MemoryStorageType")
      .value("kHost", nvidia::gxf::MemoryStorageType::kHost)
      .value("kDevice", nvidia::gxf::MemoryStorageType::kDevice)
      .value("kSystem", nvidia::gxf::MemoryStorageType::kSystem);

  pybind11::class_<nvidia::gxf::TensorDescription>(m, "TensorDescription")
      .def(pybind11::init([](std::string name, nvidia::gxf::MemoryStorageType storage_type,
                             nvidia::gxf::Shape shape, nvidia::gxf::PrimitiveType element_type,
                             uint64_t bytes_per_element, std::vector<uint64_t> strides) {
             if (strides.size() == 0) {
               return std::unique_ptr<nvidia::gxf::TensorDescription>(
                   new nvidia::gxf::TensorDescription{name, storage_type, shape, element_type,
                                                      bytes_per_element});
             } else {
               std::array<uint64_t, nvidia::gxf::Shape::kMaxRank> strides_array;
               std::copy_n(strides.begin(), strides.size(), strides_array.begin());
               return std::unique_ptr<nvidia::gxf::TensorDescription>(
                   new nvidia::gxf::TensorDescription{name, storage_type, shape, element_type,
                                                      bytes_per_element, strides_array});
             }
           }),
           "Description of nvidia::gxf::Tensor", pybind11::arg("name"),
           pybind11::arg("storage_type"), pybind11::arg("shape"), pybind11::arg("element_type"),
           pybind11::arg("bytes_per_element"), pybind11::arg("strides") = std::vector<uint64_t>{},
           pybind11::return_value_policy::reference)
      .def_readwrite("name", &nvidia::gxf::TensorDescription::name)
      .def_readwrite("storage_type", &nvidia::gxf::TensorDescription::storage_type)
      .def_readwrite("shape", &nvidia::gxf::TensorDescription::shape)
      .def_readwrite("element_type", &nvidia::gxf::TensorDescription::element_type)
      .def_readwrite("bytes_per_element", &nvidia::gxf::TensorDescription::bytes_per_element)
      .def_property(
          "strides",
          [](nvidia::gxf::TensorDescription& t) {
            if (t.strides) {
              return t.strides.value();
            } else {
              return std::array<uint64_t, nvidia::gxf::Shape::kMaxRank>{};
            }
          },
          [](nvidia::gxf::TensorDescription& t, std::vector<uint64_t> strides_) {
            throw pybind11::value_error("Setting stride not supported yet from python");
          });

  pybind11::class_<nvidia::gxf::Shape>(m, "Shape")
      .def(pybind11::init())
      .def(pybind11::init([](std::vector<int32_t> dims) {
             std::array<int32_t, nvidia::gxf::Shape::kMaxRank> dims_array;
             std::copy_n(dims.begin(), dims.size(), dims_array.begin());
             return std::unique_ptr<nvidia::gxf::Shape>(
                 new nvidia::gxf::Shape(dims_array, dims.size()));
           }),
           pybind11::return_value_policy::reference)
      .def("rank", &nvidia::gxf::Shape::rank)
      .def("size", &nvidia::gxf::Shape::size)
      .def("dimension", &nvidia::gxf::Shape::dimension)
      .def(pybind11::self == pybind11::self);

  pybind11::class_<PyLazyDLManagedTensorDeleter, std::shared_ptr<PyLazyDLManagedTensorDeleter>>(
    m, "_PyLazyDLManagedTensorDeleter").def(pybind11::init<>());

}
