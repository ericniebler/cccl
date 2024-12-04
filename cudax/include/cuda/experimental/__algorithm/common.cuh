//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ALGORITHM_COMMON
#define __CUDAX_ALGORITHM_COMMON

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <cuda/experimental/__async/async_transform.cuh>
#include <cuda/experimental/__async/tasks.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{
template <typename _Tp, typename _Decayed = _CUDA_VSTD::decay_t<_Tp>>
using __as_mdspan_t =
  _CUDA_VSTD::mdspan<typename _Decayed::value_type,
                     typename _Decayed::extents_type,
                     typename _Decayed::layout_type,
                     typename _Decayed::accessor_type>;

// clang-format off
template <typename _Tp>
_CCCL_CONCEPT __convertible_to_mdspan =
  _CCCL_REQUIRES_EXPR((_Tp))
  (
    requires(_CUDA_VSTD::convertible_to<_Tp, __as_mdspan_t<_Tp>>)
  );

template <typename _Tp>
_CCCL_CONCEPT __valid_nd_copy_fill_argument =
  _CCCL_REQUIRES_EXPR((_Tp))
  (
    requires(__convertible_to_mdspan<__relocatable_value_result_t<__async_transform_result_t<_Tp>>>)
  );
// clang-format on

struct __transform_call_fn
{
  template <typename _Fn, typename... _Args>
  _CUDAX_HOST_API decltype(auto) operator()(_Fn&& __fn, stream_ref __stream, _Args&&... __args) const
  {
    return _CUDA_VSTD::forward<_Fn>(__fn)(
      relocatable_value(async_transform(__stream, _CUDA_VSTD::forward<_Args>(__args)))...);
  }
};

inline constexpr __transform_call_fn __transform_call{};

} // namespace cuda::experimental

#endif //__CUDAX_ALGORITHM_COMMON
