//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_STREAM_LAUNCH
#define __CUDAX__EXECUTION_STREAM_LAUNCH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/experimental/__stream/stream.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <>
struct stream_domain::__apply_t<__kernel_t>
{
  // struct __fn
  // {
  //   template <class _Config, class _Fn, class... _Args>
  //   _CCCL_HOST_API auto operator()(stream& __stream, _Config __config, _Fn __fn, _Args... __args)
  //   {}
  // };

  template <class _Params>
  _CCCL_API auto operator()(stream& __stream, _Params&& __tupl) const
  {
    // return __tupl.__apply(__fn{}, static_cast<_Params&&>(__tupl), __stream);
  }
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_STREAM_LAUNCH
