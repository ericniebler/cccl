//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_DEVICE_MEMORY
#define __CUDAX_ASYNC_DETAIL_DEVICE_MEMORY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/cstddef> // for byte
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__async/device_buffer.cuh>
#include <cuda/experimental/__async/future.cuh>

namespace cuda::experimental::async
{
// An asynchronous source of device memory allocations
struct device_memory
{
private:
  struct __deleter;
  template <class _Type>
  struct new_device_buffer;

public:
  template <class _Type = _CUDA_VSTD_NOVERSION::byte>
  using buffer = device_buffer<_Type>;

  device_memory() = default;

  template <class _Type = _CUDA_VSTD_NOVERSION::byte>
  _CCCL_NODISCARD auto new_buffer(size_t count) noexcept -> new_device_buffer<_Type>;
};

struct device_memory::__deleter
{
  void operator()(const void* __pv, stream_ref __stream) const noexcept
  {
    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync, //
      "Failed trying to free device memory",
      const_cast<void*>(__pv),
      __stream.get());
  }
};

template <class _Type>
struct device_memory::new_device_buffer
{
  auto __enqueue(stream_ref stream) -> device_buffer<_Type>;

private:
  friend device_memory;

  explicit new_device_buffer(size_t count) noexcept
      : count_(count)
  {}

  size_t count_;
};

template <class _Type>
auto device_memory::new_device_buffer<_Type>::__enqueue(stream_ref stream) -> device_buffer<_Type>
{
  void* __pv = nullptr;
  _CCCL_TRY_CUDA_API(
    ::cudaMallocAsync,
    "Failed trying to asynchronously allocate device memory",
    &__pv,
    count_ * sizeof(_Type),
    stream.get());
  return {static_cast<_Type*>(__pv), count_, stream, __deleter{}};
}

template <class _Type>
_CCCL_NODISCARD auto device_memory::new_buffer(size_t count) noexcept -> new_device_buffer<_Type>
{
  return new_device_buffer<_Type>{count};
}

} // namespace cuda::experimental::async

#endif // __CUDAX_ASYNC_DETAIL_DEVICE_MEMORY
