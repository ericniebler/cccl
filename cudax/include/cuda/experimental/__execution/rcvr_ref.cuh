//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_RCVR_REF
#define __CUDAX_ASYNC_DETAIL_RCVR_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Rcvr, class _Env = env_of_t<_Rcvr>>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_ref
{
  using receiver_concept _CCCL_NODEBUG_ALIAS = receiver_t;
  _Rcvr& __rcvr_;

  template <class... _As>
  _CCCL_TRIVIAL_API void set_value(_As&&... __as) noexcept
  {
    static_cast<_Rcvr&&>(__rcvr_).set_value(static_cast<_As&&>(__as)...);
  }

  template <class _Error>
  _CCCL_TRIVIAL_API void set_error(_Error&& __err) noexcept
  {
    static_cast<_Rcvr&&>(__rcvr_).set_error(static_cast<_Error&&>(__err));
  }

  _CCCL_TRIVIAL_API void set_stopped() noexcept
  {
    static_cast<_Rcvr&&>(__rcvr_).set_stopped();
  }

  _CCCL_API auto get_env() const noexcept -> _Env
  {
    return execution::get_env(__rcvr_);
  }
};

template <class _Rcvr>
__rcvr_ref(_Rcvr&) -> __rcvr_ref<_Rcvr>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_RCVR_REF
