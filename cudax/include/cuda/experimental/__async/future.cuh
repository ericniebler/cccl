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

#include "cuda/std/__cccl/dialect.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__utility/as_const.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__async/future_fwd.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__launch/launch_transform.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

// #if !defined(_CCCL_NO_COROUTINES)
// #  include <coroutine> // IWYU pragma: keep
// #endif // !_CCCL_NO_COROUTINES
#include <memory> // IWYU pragma: keep for std::shared_ptr

namespace cuda::experimental
{
inline _CUDAX_HOST_API void __maybe_stream_wait(stream_ref __stream)
{
  if (__stream.get() != detail::__invalid_stream)
  {
    __stream.wait();
  }
}

inline _CUDAX_HOST_API stream_ref __maybe_sync_streams(stream_ref __self, stream_ref __other)
{
  if (__self.get() != detail::__invalid_stream && __other.get() != detail::__invalid_stream)
  {
    __other.wait(__self);
  }
  return __other;
}

// synchronize the launch stream with the stream on which this work was
// enqueued if they are different.
template <class _CvValue>
struct __future_transform_box : detail::__immovable
{
  using __as_kernel_arg = detail::__as_copy_arg_t<_CvValue>;

  _CUDAX_HOST_API __future_transform_box(_CvValue&& __val, stream_ref __self, stream_ref __other)
      : __self_(__self)
      , __other_(__cudax::__maybe_sync_streams(__self_, __other))
      , __inner_(detail::__launch_transform(__other, _CUDA_VSTD::forward<_CvValue>(__val)))
  {}

  _CUDAX_HOST_API ~__future_transform_box()
  {
    __cudax::__maybe_sync_streams(__other_, __self_);
  }

  _CUDAX_HOST_API operator __as_kernel_arg() &&
  {
    return _CUDA_VSTD::move(__inner_);
  }

  stream_ref __self_;
  stream_ref __other_;
  detail::__launch_transform_result_t<_CvValue> __inner_;
};

template <class _CvValue>
__future_transform_box(_CvValue&&, stream_ref, stream_ref) -> __future_transform_box<_CvValue>;

struct __future_base
{
  _CUDAX_HOST_API explicit __future_base(stream_ref __stream) noexcept
      : __stream_(__stream)
  {}

  _CUDAX_TRIVIAL_HOST_API stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

  _CUDAX_HOST_API bool ready() const noexcept
  {
    return __stream_.get() == detail::__invalid_stream;
  }

private:
  friend struct __future_access;

  template <class _Value>
  friend struct future;

  _CUDAX_HOST_API void __ready_or_wait()
  {
    __cudax::__maybe_stream_wait(_CUDA_VSTD::exchange(__stream_, stream_ref(detail::__invalid_stream)));
  }

  stream_ref __stream_; // the stream on which this work was enqueued.
};

template <class _Future>
struct __basic_future : __future_base
{
  using __future_base::__future_base;

private:
  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(_Future&& __self)
  {
    return __unpack(_CUDA_VSTD::move(__self).get_unsynchronized());
  }

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(_Future& __self)
  {
    return __unpack(__self.get_unsynchronized());
  }

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(_Future const& __self)
  {
    return __unpack(__self.get_unsynchronized());
  }

  // These __cudax_launch_transform overloads cause __launch_stream to be
  // synchronized, prior to the kernel launch, with the stream on which this
  // work was enqueued if they are different. It also causes the future's stream
  // to synchronize with __launch_stream after the kernel launch.
  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, _Future&& __self)
  {
    return __future_transform_box(_CUDA_VSTD::move(__self.get_unsynchronized()), __self.get_stream(), __launch_stream);
  }

  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, _Future& __self)
  {
    return __future_transform_box(__self.get_unsynchronized(), __self.get_stream(), __launch_stream);
  }

  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, _Future const& __self)
  {
    return __future_transform_box(__self.get_unsynchronized(), __self.get_stream(), __launch_stream);
  }
};

template <class _CvValue>
struct future_ref : __basic_future<future_ref<_CvValue>>
{
  using value_type _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_CvValue>;
  using __cv_future_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::__maybe_const<_CUDA_VSTD::is_const_v<_CvValue>, future<value_type>>;

  _CUDAX_HOST_API future_ref(__cv_future_t& __fut) noexcept
      : __basic_future<future_ref>(__fut.get_stream())
      , __value_ptr_(_CUDA_VSTD::addressof(__fut.__value_))
  {}

  _CUDAX_HOST_API future_ref(_CvValue& __val) noexcept
      : __basic_future<future_ref>(detail::__invalid_stream)
      , __value_ptr_(_CUDA_VSTD::addressof(__val))
  {}

  _CUDAX_HOST_API auto get_unsynchronized() const noexcept -> _CvValue&
  {
    return *__value_ptr_;
  }

  _CUDAX_HOST_API auto operator->() const noexcept -> const _CvValue&
  {
    return *__value_ptr_;
  }

private:
  _CvValue* __value_ptr_;
};

/*
  struct promise_type
  {
    static ::std::suspend_always initial_suspend() noexcept
    {
      return {};
    }
    static ::std::suspend_always final_suspend() noexcept
    {
      return {};
    }
    task get_return_object()
    {
      return {};
    }
    void return_void() {}
    void unhandled_exception()
    {
      throw;
    }
  };
//*/

///
/// future
///
template <class _Value>
struct future : __basic_future<future<_Value>>
{
  future(future&&)            = default;
  future& operator=(future&&) = default;

  future(future const&)            = delete;
  future& operator=(future const&) = delete;

  // Convert from another future type:
  _CCCL_TEMPLATE(class _OtherValue)
  _CCCL_REQUIRES(_CUDA_VSTD::convertible_to<_OtherValue, _Value>)
  _CUDAX_HOST_API future(future<_OtherValue>&& __other)
      : __basic_future<future>(__other.get_stream())
      , __value_(_CUDA_VSTD::move(__other.__value_))
  {}

  // Convert from a future of a range to a future of a span
  _CCCL_TEMPLATE(class _Range, class _Span = _Value)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_std_span<_Span> _CCCL_AND
                   _CUDA_VSTD::__span_compatible_range<_Range&, typename _Span::element_type>)
  _CUDAX_HOST_API future(future<_Range>& __other)
      : __basic_future<future>(__other.get_stream())
      , __value_(__other.__value_)
  {}

  _CCCL_TEMPLATE(class _Range, class _Span = _Value)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_std_span<_Span> _CCCL_AND
                   _CUDA_VSTD::__span_compatible_range<const _Range&, typename _Span::element_type>)
  _CUDAX_HOST_API future(const future<_Range>& __other)
      : __basic_future<future>(__other.get_stream())
      , __value_(__other.__value_)
  {}

  // Make a ready future:
  _CUDAX_HOST_API future(_Value&& __val)
      : __basic_future<future>(stream_ref(detail::__invalid_stream))
      , __value_(_CUDA_VSTD::move(__val))
  {}

  // Convert from a range to a ready future of a span
  _CCCL_TEMPLATE(class _Range, class _Span = _Value)
  _CCCL_REQUIRES((!__is_cudax_future<_Range>) _CCCL_AND _CUDA_VSTD::__is_std_span<_Span> _CCCL_AND
                   _CUDA_VSTD::__span_compatible_range<_Range, typename _Span::element_type>)
  _CUDAX_HOST_API future(_Range&& __other)
      : __basic_future<future>(stream_ref(detail::__invalid_stream))
      , __value_(_CUDA_VSTD::forward<_Range>(__other.__value_))
  {}

  _CUDAX_HOST_API auto get_unsynchronized() && noexcept -> _Value&&
  {
    return _CUDA_VSTD::move(__value_);
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
  template <class>
  friend struct future;
  template <class>
  friend struct future_ref;

  template <class _Action>
  _CUDAX_HOST_API explicit future(_Action __action, stream_ref __stream)
      : __basic_future<future>(__stream)
      , __value_(_CUDA_VSTD::move(__action).__enqueue(__stream))
  {}

  _Value __value_; // the async_value for the enqueued work.
};

// ///
// /// shared_future
// ///
// template <class _Value>
// struct shared_future
// {
//   _CUDAX_HOST_API stream_ref get_stream() const noexcept
//   {
//     return __fut_->__stream_;
//   }

//   _CUDAX_TRIVIAL_HOST_API bool ready() const noexcept
//   {
//     return __fut_->ready();
//   }

//   _CUDAX_TRIVIAL_HOST_API auto get_unsynchronized() const noexcept -> _Value const&
//   {
//     return __fut_->get_unsynchronized();
//   }

//   _CUDAX_TRIVIAL_HOST_API auto operator->() const noexcept -> _Value const&
//   {
//     return __fut_->operator->();
//   }

// private:
//   friend shared_future shared<>(future<_Value>&&);

//   _CUDAX_HOST_API shared_future(future<_Value>&& other)
//       : ::std::shared_ptr<future<_Value>>(_CUDA_VSTD::move(other))
//   {}

//   friend _CUDAX_HOST_API auto __cudax_unpack_future(const shared_future& __self)
//   {
//     return __unpack(_CUDA_VSTD::as_const(*__self.__fut_));
//   }

//   friend _CUDAX_TRIVIAL_HOST_API decltype(auto)
//   __cudax_launch_transform(stream_ref __launch_stream, shared_future const& __self)
//   {
//     return detail::__launch_transform(__launch_stream, *__self.__fut_);
//   }

//   _CUDAX_TRIVIAL_HOST_API void __ready_or_wait()
//   {
//     __future_access::__ready_or_wait(*__fut_);
//   }

//   ::std::shared_ptr<future<_Value>> __fut_;
// };

// template <class _Value>
// _CUDAX_HOST_API shared_future<_Value> shared(future<_Value>&& __fut)
// {
//   return shared_future<_Value>(_CUDA_VSTD::move(__fut));
// }

_CCCL_TEMPLATE(class _Future)
_CCCL_REQUIRES(__is_cudax_future<_Future>)
_CUDAX_HOST_API auto wait(_Future __fut) -> __unpack_result_t<_Future>
{
  __future_access::__ready_or_wait(__fut);
  return __unpack(_CUDA_VSTD::move(__fut));
}

_CCCL_TEMPLATE(class... _Futures)
_CCCL_REQUIRES((__is_cudax_future<_Futures> && ...))
_CUDAX_HOST_API auto wait_all(_Futures... __futs) -> _CUDA_VSTD::tuple<__unpack_result_t<_Futures>...>
{
  (__future_access::__ready_or_wait(__futs), ...);
  return {__unpack(_CUDA_VSTD::move(__futs))...};
}

struct __cref_fn
{
  template <class _Value>
  _CUDAX_HOST_API auto operator()(future<_Value> const& __fut) const noexcept
  {
    return future_ref<_Value const>{__fut};
  }

  template <class _Value>
  auto operator()(future<_Value> const&&) const = delete;
};

_CCCL_GLOBAL_CONSTANT __cref_fn cref{};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_FUTURE
