//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FUTURE
#define __CUDAX_ASYNC_DETAIL_FUTURE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/utility>

#include <cuda/experimental/__async/basic_future.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>

namespace cuda::experimental
{
///
/// future
///
template <class _Value>
struct future : __basic_future<future<_Value>>
{
  static_assert(!Future<_Value>, "Value type must not be a future");
  using value_type _CCCL_NODEBUG_ALIAS = _Value;

  future(future&&)            = default;
  future& operator=(future&&) = default;

  future(future const&)            = delete;
  future& operator=(future const&) = delete;

  // Make a ready future:
  _CUDAX_HOST_API future(_CUDA_VSTD::type_identity_t<_Value>&& __val)
      : __basic_future<future>(stream_ref(detail::__invalid_stream))
      , __value_(_CUDA_VSTD::move(__val))
  {}

  _CUDAX_HOST_API auto get_unsynchronized() && noexcept -> _Value&&
  {
    return _CUDA_VSTD::forward<_Value>(__value_);
  }

  _CUDAX_HOST_API auto get_unsynchronized() & noexcept -> _Value&
  {
    return __value_;
  }

  _CUDAX_HOST_API auto get_unsynchronized() const& noexcept -> _Value const&
  {
    return __value_;
  }

  _CUDAX_HOST_API auto operator->() const noexcept -> _Value const&
  {
    return __value_;
  }

private:
  friend struct __future_access;

  _CCCL_TEMPLATE(class _Task)
  _CCCL_REQUIRES(Task<_Task>)
  _CUDAX_HOST_API explicit future(_Task __task, stream_ref __stream)
      : __basic_future<future>(__stream)
      , __value_(_CUDA_VSTD::move(__task).enqueue(__stream))
  {}

  _CUDAX_HOST_API explicit future(_Value&& __val, stream_ref __stream)
      : __basic_future<future>(__stream)
      , __value_(_CUDA_VSTD::forward<_Value>(__val))
  {}

  _Value __value_; // the async_value for the enqueued work.
};

_CCCL_TEMPLATE(class _Value)
_CCCL_REQUIRES((!Future<_Value>) )
future(_Value&&) -> future<_Value>;

template <>
struct future<void> : __basic_future<future<void>>
{
  using value_type _CCCL_NODEBUG_ALIAS = void;

  future(future&&)            = default;
  future& operator=(future&&) = default;

  future(future const&)            = delete;
  future& operator=(future const&) = delete;

  // Make a ready future:
  _CUDAX_HOST_API future() noexcept
      : __basic_future<future>(stream_ref(detail::__invalid_stream))
  {}

  _CUDAX_HOST_API void get_unsynchronized() const noexcept {}

private:
  friend struct __future_access;

  _CCCL_TEMPLATE(class _Task)
  _CCCL_REQUIRES(Task<_Task>)
  _CUDAX_HOST_API explicit future(_Task __task, stream_ref __stream)
      : __basic_future<future>(__stream)
  {
    _CUDA_VSTD::move(__task).enqueue(__stream);
  }

  _CUDAX_HOST_API explicit future(stream_ref __stream)
      : __basic_future<future>(__stream)
  {}
};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_FUTURE
