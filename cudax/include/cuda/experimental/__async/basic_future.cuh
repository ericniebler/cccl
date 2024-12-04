//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_BASIC_FUTURE
#define __CUDAX_ASYNC_DETAIL_BASIC_FUTURE

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
#include <cuda/experimental/__async/transforms.cuh>
#include <cuda/experimental/__detail/utility.cuh>

namespace cuda::experimental
{
//! @brief This is the result of calling `launch_transform` on a `future` or a
//! `ref`. It synchronizes the launch stream with the stream on which this work
//! was enqueued if they are different.
template <class _CvValue>
struct __future_transform : __immovable
{
  _CUDAX_HOST_API __future_transform(_CvValue&& __val, stream_ref __self, stream_ref __other)
      : __self_(__self)
      , __other_(__cudax::__maybe_sync_streams(__self_, __other))
      , __inner_(async_transform(__other, _CUDA_VSTD::forward<_CvValue>(__val)))
  {}

  _CUDAX_HOST_API ~__future_transform()
  {
    __cudax::__maybe_sync_streams(__other_, __self_);
  }

  _CUDAX_HOST_API decltype(auto) relocatable_value() &&
  {
    return __cudax::relocatable_value(_CUDA_VSTD::move(__inner_));
  }

  stream_ref __self_;
  stream_ref __other_;
  __async_transform_result_t<_CvValue> __inner_;
};

template <class _CvValue>
__future_transform(_CvValue&&, stream_ref, stream_ref) -> __future_transform<_CvValue>;

//! @brief Base class for all future types. Contains a stream_ref. If the
//! stream_ref is invalid (equal to detail::__invalid_stream), the future is
//! ready.
struct __future_base
{
  _CUDAX_TRIVIAL_HOST_API explicit __future_base(stream_ref __stream) noexcept
      : __stream_(__stream)
  {}

  //! @brief Returns the stream on which this work was enqueued.
  //! @pre `valid() ==  true`
  _CUDAX_TRIVIAL_HOST_API stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

  //! @brief Returns `true` if the future refers to work in-flight; otherwise
  //! `false`.
  //!
  //! @remark A future is in-flight if:
  //! @li it refers to work scheduled on a stream and `wait()` has not yet been
  //! called on it, or
  //! @li it refers to an immediate value.
  _CUDAX_HOST_API bool valid() const noexcept
  {
    return __stream_.get() != detail::__invalid_stream;
  }

  //! @brief If `valid() ==  false`, no-op; otherwise, calls
  //! cudaStreamSynchronize on the result of `get_stream()`.
  //! @post `valid() == false`
  //! @remark This function is called from `cudax::wait` and `cudax::wait_all`.
  _CUDAX_HOST_API void wait() const
  {
    __cudax::__maybe_stream_wait(_CUDA_VSTD::exchange(__stream_, stream_ref(detail::__invalid_stream)));
  }

private:
  friend struct __future_access;

  template <class _Value>
  friend struct future;

  mutable stream_ref __stream_; // the stream on which this work was enqueued.
};

// A future<tuple<...>> will have an .unpack() member to turn it into
// a tuple of futures, suitable for structured bindings.
namespace detail
{
template <class... _Values>
inline auto __mk_repack_fn(stream_ref __stream)
{
  return [__stream](_Values&... __values) {
    return _CUDA_VSTD::tuple<future<_Values>...>{
      __future_access::__make_future_from_value<_Values>(_CUDA_VSTD::forward<_Values>(__values), __stream)...};
  };
}

template <class _Future>
struct __future_tuple_base;

template <class... _Values>
struct __future_tuple_base<future<_CUDA_VSTD::tuple<_Values...>>> : __future_base
{
  using __future_base::__future_base;

  _CUDAX_HOST_API auto unpack() && -> _CUDA_VSTD::tuple<future<_Values>...>
  {
    using __future_t = future<_CUDA_VSTD::tuple<_Values...>>;
    auto& __tupl     = static_cast<__future_t&>(*this).get_unsynchronized();
    return _CUDA_VSTD::apply(detail::__mk_repack_fn<_Values...>(get_stream()), __tupl);
  }
};

template <class _Is, class... _Values>
struct __future_tuple_base<future<__async::__tupl<_Is, _Values...>>> : __future_base
{
  using __future_base::__future_base;

  _CUDAX_HOST_API auto unpack() && -> _CUDA_VSTD::tuple<future<_Values>...>
  {
    using __future_t = future<__async::__tupl<_Is, _Values...>>;
    auto& __tupl     = static_cast<__future_t&>(*this).get_unsynchronized();
    return __tupl.__apply(detail::__mk_repack_fn<_Values...>(get_stream()), __tupl);
  }
};

// clang-format off
template <class _Future>
_CCCL_CONCEPT __is_future_tuple =
  _CCCL_REQUIRES_EXPR((_Future))
  (
    typename(typename __future_tuple_base<_Future>::__future_base)
  );
// clang-format on

template <class _Future>
using __future_base_t = _CUDA_VSTD::_If<__is_future_tuple<_Future>, __future_tuple_base<_Future>, __future_base>;
} // namespace detail

//! @brief CRTP base for all future types. Contains a stream_ref. It provides
//! customizations for the `get_unsynchronized` and `async_transform` customization
//! points. Both `future` and `ref` derive from this class.
template <class _Future>
struct __basic_future : detail::__future_base_t<_Future>
{
  _CUDAX_TRIVIAL_HOST_API explicit __basic_future(stream_ref __stream) noexcept
      : detail::__future_base_t<_Future>(__stream)
  {}

private:
  // Return unsynchronized access to the underlying value of a future.
  _CCCL_TEMPLATE(class _Self)
  _CCCL_REQUIRES(__decays_to_derived_from<_Self, _Future>)
  friend _CUDAX_TRIVIAL_HOST_API decltype(auto) cuda_get_unsynchronized(_Self&& __self)
  {
    return get_unsynchronized(_CUDA_VSTD::forward<_Self>(__self).get_unsynchronized());
  }

  // This cuda_async_transform overload causes __launch_stream to be
  // synchronized, prior to the kernel launch, with the stream on which this
  // work was enqueued -- but only if they are different. It also causes the
  // future's stream to synchronize with __launch_stream after the kernel
  // launch.
  _CCCL_TEMPLATE(class _Self)
  _CCCL_REQUIRES(__decays_to_derived_from<_Self, _Future>)
  friend _CUDAX_TRIVIAL_HOST_API auto cuda_async_transform(stream_ref __launch_stream, _Self&& __self)
  {
    return __future_transform(
      _CUDA_VSTD::forward<_Self>(__self).get_unsynchronized(), __self.get_stream(), __launch_stream);
  }
};

//! @brief Future types can be used with the stream_ref operator<< to enqueue
//! tasks on the stream.
_CCCL_TEMPLATE(class _Future, class _Task)
_CCCL_REQUIRES(Future<_Future> _CCCL_AND Task<_Task>)
_CUDAX_HOST_API decltype(auto) operator<<(_Future&& __fut, _Task __task)
{
  return __fut.get_stream() << _CUDA_VSTD::move(__task);
}

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_BASIC_FUTURE
