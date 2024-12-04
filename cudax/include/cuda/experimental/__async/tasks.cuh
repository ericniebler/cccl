//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_TASKS
#define __CUDAX_ASYNC_DETAIL_TASKS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/utility>

#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>

namespace cuda::experimental
{
// If the argument is an task, enqueue it and return the result. Otherwise,
// return the argument as-is.
struct __maybe_enqueue_fn
{
  template <class _Value>
  _CUDAX_HOST_API decltype(auto) operator()(_Value&& __value, [[maybe_unused]] stream_ref __stream) const
  {
    if constexpr (Task<_Value>)
    {
      return _CUDA_VSTD::forward<_Value>(__value).enqueue(__stream);
    }
    else
    {
      return _Value(_CUDA_VSTD::forward<_Value>(__value));
    }
  }
};

inline namespace __cpo
{
inline constexpr __maybe_enqueue_fn __maybe_enqueue{};
} // namespace __cpo

template <typename _Tp>
_CCCL_CONCEPT async_param = bool(_CUDA_VSTD::is_object_v<_Tp> || Task<_Tp> || __cudax_ref<_Tp>);

template <typename _Tp, typename _Value>
_CCCL_CONCEPT async_param_of = async_param<_Tp> && _CUDA_VSTD::convertible_to<async_result_of_t<_Tp>, _Value>;

struct __async_call
{
  _CCCL_TEMPLATE(typename _Fn, typename... _Args)
  _CCCL_REQUIRES((async_param<_Args> && ...))
  _CUDAX_HOST_API decltype(auto) operator()(_Fn&& __fn, stream_ref __stream, _Args&&... __args) const
  {
    return _CUDA_VSTD::forward<_Fn>(__fn)(__maybe_enqueue(_CUDA_VSTD::forward<_Args>(__args), __stream)...);
  }
};

inline constexpr __async_call async_call{};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_TASKS
