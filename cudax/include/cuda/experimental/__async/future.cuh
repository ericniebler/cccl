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

#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/utility>

#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__launch/launch_transform.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <memory> // for std::shared_ptr

namespace cuda::experimental
{
template <class _Value>
struct _CCCL_NODISCARD future;

template <class _Value>
struct _CCCL_NODISCARD shared_future;

template <class _Value>
_CUDAX_HOST_API shared_future<_Value> shared(future<_Value>&& __fut);

void __cudax_unpack_future();

template <class _Future>
using __raw_unpack_result_t = decltype(__cudax_unpack_future(_CUDA_VSTD::declval<_Future>()));

struct __unpack_fn
{
  template <class _Future>
  _CUDAX_TRIVIAL_HOST_API decltype(auto) operator()(_Future&& __fut) const
  {
    if constexpr (_CUDA_VSTD::_IsValidExpansion<__raw_unpack_result_t, _Future>::value)
    {
      return __cudax_unpack_future(static_cast<_Future&&>(__fut));
    }
    else
    {
      return _Future(static_cast<_Future&&>(__fut));
    }
  }
};

inline constexpr __unpack_fn __unpack{};

template <class _Future>
using __unpack_result_t = decltype(__unpack(_CUDA_VSTD::declval<_Future>()));

template <class _Value>
using __action_result_t = decltype(_CUDA_VSTD::declval<_Value>().__enqueue(cuda::stream_ref()));

template <class _Value>
_CUDAX_HOST_API void __test_cudax_future(const future<_Value>&);

template <class _Value>
_CUDAX_HOST_API void __test_cudax_future(const shared_future<_Value>&);

// clang-format off
template <class _Type>
_CCCL_CONCEPT __is_cudax_future =
  _CCCL_REQUIRES_EXPR((_Type), const _Type& __value)
  (
    (::cuda::experimental::__test_cudax_future(__value))
  );

template <class _Type>
_CCCL_CONCEPT __is_cudax_action =
  _CCCL_REQUIRES_EXPR((_Type), _Type&& __value, cuda::stream_ref __stream)
  (
    (_CUDA_VSTD::move(__value).__enqueue(__stream))
  );
// clang-format on

template <class _Action>
_CUDAX_TRIVIAL_HOST_API auto __make_future(_Action __action, stream_ref __stream)
{
  return future<__action_result_t<_Action>>(_CUDA_VSTD::move(__action), __stream);
}

///
/// future
///
template <class _Value>
struct _CCCL_NODISCARD future
{
  future(future&&)            = default;
  future& operator=(future&&) = default;

  _CUDAX_HOST_API stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

private:
  template <class _Action>
  friend auto __make_future(_Action, stream_ref);

  template <bool _IsConst>
  struct __transform;

  template <class _Action>
  _CUDAX_HOST_API explicit future(_Action __action, stream_ref __stream)
      : __value_(_CUDA_VSTD::move(__action).__enqueue(__stream))
      , __stream_(__stream)
  {}

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(future& __self)
  {
    return __unpack(__self.__value_);
  }

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(future&& __self)
  {
    return __unpack(_CUDA_VSTD::move(__self).__value_);
  }

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(future* __self)
  {
    return __unpack(__self->__value_);
  }

  // These __cudax_launch_transform overloads cause __launch_stream to be
  // synchronized, prior to the kernel launch, with the stream on which this
  // work was enqueued if they are different. It also causes the future's stream
  // to synchronize with __launch_stream after the kernel launch.
  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, future& __self)
  {
    return __transform<false>(__self, __launch_stream);
  }

  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, const future& __self)
  {
    return __transform<true>(__self, __launch_stream);
  }

  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, future* __self)
  {
    return __transform<false>(*__self, __launch_stream);
  }

  friend _CUDAX_TRIVIAL_HOST_API auto __cudax_launch_transform(stream_ref __launch_stream, const future* __self)
  {
    return __transform<true>(*__self, __launch_stream);
  }

  _Value __value_; // the async_value for the enqueued work.
  stream_ref __stream_; // the stream on which this work was enqueued.
};

// synchronize the launch __stream with the __stream on which this work was
// enqueued if they are different.
template <class _Value>
template <bool _IsConst>
struct future<_Value>::__transform : detail::__immovable
{
  using __cv_async_value = _CUDA_VSTD::__maybe_const<_IsConst, _Value>;
  using __as_kernel_arg  = detail::__as_copy_arg_t<__cv_async_value&>;

  __transform(_CUDA_VSTD::__maybe_const<_IsConst, future>& __self, stream_ref __other)
      : __self_(__self.__stream_)
      , __other_((__other.wait(__self_), __other))
      , __inner_(detail::__launch_transform(__other, __self.__value_))
  {}

  ~__transform()
  {
    __self_.wait(__other_);
  }

  operator __as_kernel_arg() &&
  {
    return _CUDA_VSTD::move(__inner_);
  }

  stream_ref __self_;
  stream_ref __other_;
  detail::__launch_transform_result_t<__cv_async_value&> __inner_;
};

///
/// shared_future
///
template <class _Value>
struct _CCCL_NODISCARD shared_future
{
  _CUDAX_HOST_API stream_ref get_stream() const noexcept
  {
    return __fut_->__stream_;
  }

  friend _CUDAX_HOST_API auto __cudax_unpack_future(shared_future& __self)
  {
    return __unpack(*__self.__fut_);
  }

private:
  friend shared_future shared<>(future<_Value>&&);

  _CUDAX_HOST_API shared_future(future<_Value>&& other)
      : ::std::shared_ptr<future<_Value>>(_CUDA_VSTD::move(other))
  {}

  friend _CUDAX_TRIVIAL_HOST_API decltype(auto)
  __cudax_launch_transform(stream_ref __launch_stream, const shared_future& __self)
  {
    return detail::__launch_transform(__launch_stream, *__self.__fut_);
  }

  ::std::shared_ptr<future<_Value>> __fut_;
};

template <class _Value>
_CUDAX_HOST_API shared_future<_Value> shared(future<_Value>&& __fut)
{
  return shared_future<_Value>(__fut);
}

_CCCL_TEMPLATE(class _Future)
_CCCL_REQUIRES(__is_cudax_future<_Future>)
_CUDAX_HOST_API auto wait(_Future __fut) -> __unpack_result_t<_Future>
{
  __fut.get_stream().wait();
  return __unpack(_CUDA_VSTD::move(__fut));
}

_CCCL_TEMPLATE(class... _Futures)
_CCCL_REQUIRES((__is_cudax_future<_Futures> && ...))
_CUDAX_HOST_API auto wait_all(_Futures... __futs) -> _CUDA_VSTD::tuple<__unpack_result_t<_Futures>...>
{
  (__futs.get_stream().wait(), ...);
  return {__unpack(_CUDA_VSTD::move(__futs))...};
}

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_FUTURE
