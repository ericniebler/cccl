//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_WAIT
#define __CUDAX_ASYNC_DETAIL_WAIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/future_access.cuh>

namespace cuda::experimental
{
struct __wait_t
{
  _CCCL_TEMPLATE(class _Future)
  _CCCL_REQUIRES(Future<_Future>)
  _CUDAX_HOST_API auto operator()(_Future __fut) const -> __wait_result_t<_Future>
  {
    __fut.wait();
    return get_unsynchronized(_CUDA_VSTD::move(__fut));
  }
};

struct __wait_all_t
{
  _CCCL_TEMPLATE(class... _Futures)
  _CCCL_REQUIRES((Future<_Futures> && ...))
  _CUDAX_HOST_API auto operator()(_Futures... __futs) const -> _CUDA_VSTD::tuple<__wait_result_t<_Futures>...>
  {
    (__futs.wait(), ...);
    return {get_unsynchronized(_CUDA_VSTD::move(__futs))...};
  }
};

inline constexpr __wait_t wait{};
inline constexpr __wait_all_t wait_all{};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_WAIT
