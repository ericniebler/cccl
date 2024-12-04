//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_STREAM_REF
#define __CUDAX_ASYNC_DETAIL_STREAM_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stream/stream_ref.cuh>

// This file is a convenience header for
// <cuda/experimental/__stream/stream_ref.cuh> that also includes the parts of
// cudax::async that are required to use the future-based async model.
#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/future_access.cuh>

namespace cuda::experimental
{
//! @brief Enqueue a unit of work into this stream, and return a future
//! to the result of that work.
_CCCL_TEMPLATE(class _Task)
_CCCL_REQUIRES(Task<_Task>)
_CCCL_NODISCARD _CUDAX_HOST_API decltype(auto) operator<<(stream_ref __stream, _Task __task)
{
  // avoid returning a future of a future:
  if constexpr (Future<__task_value_of<_Task>>)
  {
    return _CUDA_VSTD::move(__task).enqueue(__stream);
  }
  // Treat __async::__tuple as special so that rather than returning a
  // future<__async::__tuple<Values>>, we return a
  // cuda::std::tuple<future<Values>...>, which is suitable as the initializer
  // of a structured binding.
  else if constexpr (__is_specialization_of<__task_value_of<_Task>, __async::__tupl>)
  {
    return __future_access::__make_future_from_task(_CUDA_VSTD::move(__task), __stream).unpack();
  }
  else
  {
    return __future_access::__make_future_from_task(_CUDA_VSTD::move(__task), __stream);
  }
}
} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_STREAM_REF
