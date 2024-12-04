//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FUTURE_FWD
#define __CUDAX_ASYNC_DETAIL_FUTURE_FWD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>

namespace cuda::experimental
{
struct stream_ref;

template <class _Value>
struct future;

template <class _CvValue>
struct future_ref;

// template <class _Value>
// struct shared_future;

// template <class _Value>
// _CUDAX_HOST_API shared_future<_Value> shared(future<_Value>&& __fut);

namespace __unpack_ns
{
void __cudax_unpack_future();

template <class _Future>
using __raw_unpack_result_t = decltype(__cudax_unpack_future(declval<_Future>()));

struct __fn
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
} // namespace __unpack_ns

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT __unpack_ns::__fn __unpack{};
} // namespace __cpo

template <class _Future>
using __unpack_result_t = decltype(__unpack(declval<_Future>()));

template <class _Value>
using __action_result_t = decltype(declval<_Value>().__enqueue(declval<stream_ref const&>()));

template <class _Value>
_CUDAX_HOST_API void __test_cudax_future(const future<_Value>&);

template <class _Value>
_CUDAX_HOST_API void __test_cudax_future(const future_ref<_Value>&);

// template <class _Value>
// _CUDAX_HOST_API void __test_cudax_future(const shared_future<_Value>&);

struct __future_access
{
  template <class _Action>
  static _CUDAX_TRIVIAL_HOST_API auto __make_future(_Action __action, stream_ref const& __stream)
  {
    return future<__action_result_t<_Action>>(_CUDA_VSTD::move(__action), __stream);
  }

  template <class _Future>
  static _CUDAX_TRIVIAL_HOST_API void __ready_or_wait(_Future& __fut)
  {
    __fut.__ready_or_wait();
  }
};

// clang-format off
template <class _Type>
_CCCL_CONCEPT __is_cudax_future =
  _CCCL_REQUIRES_EXPR((_Type), const _Type& __value)
  (
    (__cudax::__test_cudax_future(__value))
  );

template <class _Type>
_CCCL_CONCEPT __is_cudax_action =
  _CCCL_REQUIRES_EXPR((_Type), _Type&& __value, stream_ref const& __stream)
  (
    (_CUDA_VSTD::move(__value).__enqueue(__stream))
  );
// clang-format on

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_FUTURE_FWD
