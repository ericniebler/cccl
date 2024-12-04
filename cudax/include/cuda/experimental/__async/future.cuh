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
struct stream_ref;
}

namespace cuda::experimental::__async
{
template <class _Value>
struct _CCCL_NODISCARD future;

template <class _Value>
struct _CCCL_NODISCARD shared_future;

template <class _Value>
_CUDAX_HOST_API shared_future<_Value> shared(future<_Value>&& fut);

void __cudax_unpack_future();

template <class _Future>
using __raw_unpack_result_t = decltype(__cudax_unpack_future(cuda::std::declval<_Future>()));

struct __unpack_fn
{
  template <class _Future>
  _CUDAX_TRIVIAL_HOST_API decltype(auto) operator()(_Future&& future) const
  {
    if constexpr (cuda::std::_IsValidExpansion<__raw_unpack_result_t, _Future>::value)
    {
      return __cudax_unpack_future(static_cast<_Future&&>(future));
    }
    else
    {
      return _Future(static_cast<_Future&&>(future));
    }
  }
};

inline constexpr __unpack_fn __unpack{};

template <class _Future>
using __unpack_result_t = decltype(__unpack(cuda::std::declval<_Future>()));

template <class _Value>
using __action_result_t = decltype(cuda::std::declval<_Value>().__enqueue(cuda::stream_ref()));

template <class _Value>
_CUDAX_HOST_API void __test_cudax_future(const future<_Value>&);

template <class _Value>
_CUDAX_HOST_API void __test_cudax_future(const shared_future<_Value>&);

// clang-format off
template <class _Type>
_CCCL_CONCEPT __is_cudax_future =
  _CCCL_REQUIRES_EXPR((_Type), const _Type& __value)
  (
    __async::__test_cudax_future(__value)
  );

template <class _Type>
_CCCL_CONCEPT __is_cudax_action =
  _CCCL_REQUIRES_EXPR((_Type), _Type&& __value, cuda::stream_ref __stream)
  (
    cuda::std::move(__value).__enqueue(__stream)
  );
// clang-format on

template <class _Action>
_CUDAX_TRIVIAL_HOST_API auto __as_future(_Action __action, stream_ref __stream) -> future<__action_result_t<_Action>>
{
  using __value_t = __action_result_t<_Action>;
  return future<__value_t>(cuda::std::move(__action), __stream);
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
  friend auto __as_future(_Action, stream_ref) -> future<__action_result_t<_Action>>;

  // synchronize the launch __stream with the __stream on which this work was
  // enqueued if they are different.
  template <bool _IsConst>
  struct __transform : detail::__immovable
  {
    using __cv_async_value = cuda::std::__maybe_const<_IsConst, _Value>;
    using __as_kernel_arg  = detail::__as_copy_arg_t<__cv_async_value&>;

    __transform(cuda::std::__maybe_const<_IsConst, future>& __self, stream_ref __other)
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
      return cuda::std::move(__inner_);
    }

    stream_ref __self_;
    stream_ref __other_;
    detail::__launch_transform_result_t<__cv_async_value&> __inner_;
  };

  template <class _Action>
  _CUDAX_HOST_API explicit future(_Action __action, stream_ref __stream)
      : __value_(cuda::std::move(__action).__enqueue(__stream))
      , __stream_(__stream)
  {}

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(future& __self)
  {
    return __unpack(__self.__value_);
  }

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(future&& __self)
  {
    return __unpack(cuda::std::move(__self).__value_);
  }

  _CUDAX_TRIVIAL_HOST_API friend decltype(auto) __cudax_unpack_future(future* __self)
  {
    return __unpack(__self->__value_);
  }

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

///
/// shared_future
///
template <class _Value>
struct _CCCL_NODISCARD shared_future
{
  _CUDAX_HOST_API stream_ref get_stream() const noexcept
  {
    return fut_->__stream_;
  }

  friend _CUDAX_HOST_API auto __cudax_unpack_future(shared_future& __self)
  {
    return __unpack(*__self.fut_);
  }

private:
  friend shared_future shared<>(future<_Value>&&);

  _CUDAX_HOST_API shared_future(future<_Value>&& other)
      : ::std::shared_ptr<future<_Value>>(cuda::std::move(other))
  {}

  friend _CUDAX_TRIVIAL_HOST_API decltype(auto)
  __cudax_launch_transform(stream_ref __launch_stream, const shared_future& __self)
  {
    return detail::__launch_transform(__launch_stream, *__self.fut_);
  }

  ::std::shared_ptr<future<_Value>> fut_;
};

template <class _Value>
_CUDAX_HOST_API shared_future<_Value> shared(future<_Value>&& fut)
{
  return shared_future<_Value>(fut);
}

_CCCL_TEMPLATE(class... Futures)
_CCCL_REQUIRES((__async::__is_cudax_future<Futures> && ...))
_CUDAX_HOST_API auto wait(Futures... futures) -> cuda::std::tuple<__async::__unpack_result_t<Futures>...>
{
  (futures.get_stream().wait(), ...);
  return {__unpack(cuda::std::move(futures))...};
}

} // namespace cuda::experimental::__async

#endif // __CUDAX_ASYNC_DETAIL_FUTURE
