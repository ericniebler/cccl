//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_FUSE__
#define __CUDAX_ASYNC_FUSE__

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/utility>

#include <cuda/experimental/__async/future.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>

namespace cuda::experimental
{

template <typename... _Tasks>
struct __fuse_task
{
  // By using __async::__tuple as the enqueue return type, we opt-in
  // for stream.insert to return a tuple of futures instead of a future
  // of a tuple.
  using __value_t = __async::__tuple<__task_value_of<_Tasks>...>;

  __value_t enqueue(const stream_ref& __stream) &&
  {
    return __tasks_.__apply(
      [__stream](_Tasks&... __tasks) {
        return __value_t{{_CUDA_VSTD::move(__tasks).enqueue(__stream)}...};
      },
      __tasks_);
  }

  __async::__tuple<_Tasks...> __tasks_;
};

struct __fuse_t
{
  _CCCL_TEMPLATE(typename... _Tasks)
  _CCCL_REQUIRES((Task<_Tasks> && ...))
  __fuse_task<_Tasks...> operator()(_Tasks... __tasks) const
  {
    return __fuse_task<_Tasks...>{{{_CUDA_VSTD::move(__tasks)}...}};
  }
};

inline constexpr __fuse_t fuse{};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_FUSE__
