//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_GRAPH_DOMAIN
#define __CUDAX__EXECUTION_GRAPH_DOMAIN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_callable.h>

#include <cuda/experimental/__execution/domain.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//////////////////////////////////////////////////////////////////////////////////////////
// graph domain
struct graph_domain : default_domain
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Tag>
  struct __apply_t;
  struct __visitor_t;
  struct __context_t;
  struct __state_t;
  struct __visit_result_t;

  // for walking a sender expression and turning it into a CUDA graph
  struct __visitor_t
  {
    template <class _Tag, class _Params, class... _Child>
    _CCCL_TRIVIAL_HOST_API auto operator()(__context_t& __context, _Tag, _Params&& __params, _Child&&... __child) const
    {
      return __apply_t<_Tag>()(__context, static_cast<_Params&&>(__params), static_cast<_Child&&>(__child)...);
    }
  };

public:
  _CCCL_TEMPLATE(class _Tag, class _Sndr, class... _Args)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_callable_v<__apply_t<_Tag>, _Sndr, _Args...>)
  _CCCL_TRIVIAL_HOST_API static constexpr auto apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(
    _CUDA_VSTD::__is_nothrow_callable_v<__apply_t<_Tag>, _Sndr, _Args...>)
    -> _CUDA_VSTD::__call_result_t<__apply_t<_Tag>, _Sndr, _Args...>
  {
    return __apply_t<_Tag>()(static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }
};

_CCCL_GLOBAL_CONSTANT graph_domain::__visitor_t __graph_visitor{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_GRAPH_DOMAIN
