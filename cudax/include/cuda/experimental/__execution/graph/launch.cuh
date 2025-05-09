//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_GRAPH_LAUNCH
#define __CUDAX__EXECUTION_GRAPH_LAUNCH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/graph/domain.cuh>
#include <cuda/experimental/__execution/graph/visit.cuh>
#include <cuda/experimental/__graph/graph.cuh>
#include <cuda/experimental/__launch/launch.cuh>

#include <cuda_runtime_api.h>

namespace cuda::experimental::execution
{
template <>
struct graph_domain::__apply_t<__kernel_t>
{
  struct __fn
  {
    template <class _Context, class _Config, class _Fn, class... _Args>
    _CCCL_HOST_API auto operator()(_Context& __context, _Config __config, _Fn __fn, _Args... __args) -> __visit_result_t
    {
      __visit_result_t __result{__context.__graph_};
      auto* __launcher = &__kernel_launcher<_Config, _Fn, _Args...>;
      auto* __pfn      = reinterpret_cast<void*>(__launcher);

      // __kernel_args_offset must be the index of the first of "__args" in the kernelParams array.
      static_assert(__kernel_args_offset == 2, "kernel_args_offset must be 2");
      void* __pvargs[] = {
        _CUDA_VSTD::addressof(__config), _CUDA_VSTD::addressof(__fn), _CUDA_VSTD::addressof(__args)...};

      // Parse the kernel configuration into a cudaLaunchConfig_t struct that we can use to
      // initialize the cudaKernelNodeParams struct.
      cudaLaunchConfig_t __launch_config{};
      _CCCL_TRY_CUDA_API(__detail::apply_kernel_config, "kernel configuration failed", __config, __launch_config, __pfn);

      const cudaKernelNodeParams __params{
        .func           = __pfn,
        .gridDim        = __config.dims.extents(thread, block),
        .blockDim       = __config.dims.extents(block, grid),
        .sharedMemBytes = static_cast<unsigned int>(__launch_config.dynamicSmemBytes),
        .kernelParams   = __pvargs,
        .extra          = nullptr,
      };

      _CCCL_TRY_CUDA_API(
        cudaGraphAddKernelNode,
        "cudaGraphAddKernelNode failed",
        &__result.__node_.__node_,
        __context.__graph_.get(),
        nullptr, // dependencies
        0, // numDependencies
        &__params);

      return __result;
    }
  };

  template <class _Context, class _Params>
  _CCCL_API auto operator()(_Context& __context, _Params&& __params) const -> __visit_result_t
  {
    return _CUDA_VSTD::__apply(__fn{}, static_cast<_Params&&>(__params), __context);
  }
};
} // namespace cuda::experimental::execution

#endif // __CUDAX__EXECUTION_GRAPH_LAUNCH
