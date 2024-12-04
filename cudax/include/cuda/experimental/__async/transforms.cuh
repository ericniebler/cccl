//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_TRANSFORMS
#define __CUDAX_ASYNC_DETAIL_TRANSFORMS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/async_transform.cuh>
#include <cuda/experimental/__async/get_unsynchronized.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>
#include <cuda/experimental/__detail/utility.cuh>

namespace cuda::experimental
{
//! @brief If the stream is invalid, no-op; otherwise, waits for the stream to
//! finish executing all work that was enqueued on the stream,
inline _CUDAX_HOST_API void __maybe_stream_wait(stream_ref __stream)
{
  if (__stream.get() != detail::__invalid_stream)
  {
    __stream.wait();
  }
}

//! @brief If either stream is invalid, no-op; otherwise, causes `__launch_stream` to
//! wait for `__self_stream` to execute all work that is currently enqueued on
//! `__self_stream`.
inline _CUDAX_HOST_API stream_ref __maybe_sync_streams(stream_ref __self_stream, stream_ref __launch_stream)
{
  if (__self_stream.get() != detail::__invalid_stream && __launch_stream.get() != detail::__invalid_stream)
  {
    __launch_stream.wait(__self_stream);
  }
  return __launch_stream;
}

//! @brief This is the result of calling `launch_transform` on a `future` or a
//! `ref`. It synchronizes the launch stream with the stream on which this work
//! was enqueued if they are different.
template <class _CvValue>
struct __stream_provider_transform : __immovable
{
  _CUDAX_HOST_API __stream_provider_transform(_CvValue&& __val, stream_ref __self_stream, stream_ref __launch_stream)
      : __self_(__self_stream)
      , __other_(__cudax::__maybe_sync_streams(__self_, __launch_stream))
      , __value_(_CUDA_VSTD::forward<_CvValue>(__val))
  {}

  _CUDAX_HOST_API ~__stream_provider_transform()
  {
    __cudax::__maybe_sync_streams(__other_, __self_);
  }

  _CUDAX_HOST_API decltype(auto) relocatable_value() &&
  {
    return __cudax::relocatable_value(_CUDA_VSTD::move(__value_));
  }

  stream_ref __self_;
  stream_ref __other_;
  _CvValue __value_;
};

template <class _CvValue>
__stream_provider_transform(_CvValue&&, stream_ref, stream_ref) -> __stream_provider_transform<_CvValue>;

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_TRANSFORMS
