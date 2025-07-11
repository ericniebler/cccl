//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_CPOS
#define __CUDAX_EXECUTION_CPOS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// make the completion tags equality comparable
template <__disposition _Disposition>
struct __completion_tag
{
  template <__disposition _OtherDisposition>
  _CCCL_TRIVIAL_API constexpr auto operator==(__completion_tag<_OtherDisposition>) const noexcept -> bool
  {
    return _Disposition == _OtherDisposition;
  }

  template <__disposition _OtherDisposition>
  _CCCL_TRIVIAL_API constexpr auto operator!=(__completion_tag<_OtherDisposition>) const noexcept -> bool
  {
    return _Disposition != _OtherDisposition;
  }

  static constexpr __disposition __disposition = _Disposition;
};

struct set_value_t : __completion_tag<__disposition::__value>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Rcvr, class... _Ts>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Rcvr&& __rcvr, _Ts&&... __ts) const noexcept
    -> decltype(static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...)), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...)));
    static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...);
  }
};

struct set_error_t : __completion_tag<__disposition::__error>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Rcvr, class _Ey>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Rcvr&& __rcvr, _Ey&& __e) const noexcept
    -> decltype(static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e)))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e))), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e))));
    static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e));
  }
};

struct set_stopped_t : __completion_tag<__disposition::__stopped>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Rcvr>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Rcvr&& __rcvr) const noexcept
    -> decltype(static_cast<_Rcvr&&>(__rcvr).set_stopped())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_stopped()), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_stopped()));
    static_cast<_Rcvr&&>(__rcvr).set_stopped();
  }
};

struct start_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _OpState>
  _CCCL_TRIVIAL_API constexpr auto operator()(_OpState& __opstate) const noexcept -> decltype(__opstate.start())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(__opstate.start()), void>);
    static_assert(noexcept(__opstate.start()));
    __opstate.start();
  }
};

// connect
struct connect_t
{
private:
  struct __transform_fn
  {
    template <class _Sndr, class _Rcvr, class _Domain = __late_domain_of_t<_Sndr, env_of_t<_Rcvr>>>
    _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr&& __sndr, _Rcvr __rcvr) const
      noexcept(noexcept(transform_sender(_Domain{}, declval<_Sndr>(), get_env(__rcvr))))
        -> decltype(transform_sender(_Domain{}, declval<_Sndr>(), get_env(__rcvr)))
    {
      return transform_sender(_Domain{}, static_cast<_Sndr&&>(__sndr), get_env(__rcvr));
    }
  };

  template <bool _HasSndrTransform>
  struct __impl_fn
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class _Sndr, class _Rcvr>
    _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr&& __sndr, _Rcvr __rcvr) const
      noexcept(noexcept(declval<_Sndr>().connect(declval<_Rcvr>())))
        -> decltype(declval<_Sndr>().connect(declval<_Rcvr>()))
    {
      return static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr));
    }
  };

  template <class _Sndr, class _Rcvr>
  using __impl_t = __impl_fn<__has_sender_transform<_Sndr, env_of_t<_Rcvr>>>;

public:
  template <class _Sndr, class _Rcvr>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr&& __sndr, _Rcvr __rcvr) const
    noexcept(_CUDA_VSTD::__is_nothrow_callable_v<__impl_t<_Sndr, _Rcvr>, _Sndr, _Rcvr>)
      -> _CUDA_VSTD::__call_result_t<__impl_t<_Sndr, _Rcvr>, _Sndr, _Rcvr>
  {
    return __impl_t<_Sndr, _Rcvr>{}(static_cast<_Sndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr));
  }
};

template <>
struct connect_t::__impl_fn<true>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr, class _Rcvr>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr&& __sndr, _Rcvr __rcvr) const
    noexcept(noexcept(__transform_fn{}(declval<_Sndr>(), __rcvr).connect(declval<_Rcvr>())))
      -> decltype(__transform_fn{}(declval<_Sndr>(), __rcvr).connect(declval<_Rcvr>()))
  {
    return __transform_fn{}(static_cast<_Sndr&&>(__sndr), __rcvr).connect(static_cast<_Rcvr&&>(__rcvr));
  }
};

struct schedule_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sch>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sch&& __sch) const noexcept
  {
    static_assert(noexcept(static_cast<_Sch&&>(__sch).schedule()));
    return static_cast<_Sch&&>(__sch).schedule();
  }
};

_CCCL_GLOBAL_CONSTANT set_value_t set_value{};
_CCCL_GLOBAL_CONSTANT set_error_t set_error{};
_CCCL_GLOBAL_CONSTANT set_stopped_t set_stopped{};
_CCCL_GLOBAL_CONSTANT start_t start{};
_CCCL_GLOBAL_CONSTANT connect_t connect{};
_CCCL_GLOBAL_CONSTANT schedule_t schedule{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_CPOS
