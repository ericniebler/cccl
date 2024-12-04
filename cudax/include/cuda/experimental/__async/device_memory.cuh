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
  template <class _Type>
  struct __make_buffer_task;

public:
  template <class _Type = _CUDA_VSTD_NOVERSION::byte>
  using buffer = device_buffer<_Type>;

  device_memory() = default;

  template <class _Type = _CUDA_VSTD_NOVERSION::byte>
  _CCCL_NODISCARD auto new_buffer(size_t count) noexcept -> __make_buffer_task<_Type>;
};

template <class _Type>
struct device_memory::__make_buffer_task
{
  _CCCL_NODISCARD auto enqueue(stream_ref stream) -> device_buffer<_Type>
  {
    return device_buffer<_Type>{count_, stream};
  }

private:
  friend device_memory;

  explicit __make_buffer_task(size_t count) noexcept
      : count_(count)
  {}

  size_t count_;
};

template <class _Type>
_CCCL_NODISCARD auto device_memory::new_buffer(size_t count) noexcept -> __make_buffer_task<_Type>
{
  return __make_buffer_task<_Type>{count};
}

} // namespace cuda::experimental::async

#endif // __CUDAX_ASYNC_DETAIL_DEVICE_MEMORY
