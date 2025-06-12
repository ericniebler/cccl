//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_BULK
#define __CUDAX_EXECUTION_STREAM_BULK

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// #include <cuda/atomic>
// #include <cuda/std/__exception/cuda_error.h>
// #include <cuda/std/__memory/addressof.h>
// #include <cuda/std/__memory/unique_ptr.h>
// #include <cuda/std/__tuple_dir/ignore.h>
// #include <cuda/std/__type_traits/is_same.h>
// #include <cuda/std/__type_traits/remove_cvref.h>
// #include <cuda/std/__type_traits/remove_reference.h>
// #include <cuda/std/__type_traits/type_list.h>
// #include <cuda/std/__utility/pod_tuple.h>
#include <cuda/std/__utility/forward_like.h>

// #include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/bulk.cuh>
// #include <cuda/experimental/__execution/domain.cuh>
// #include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
// #include <cuda/experimental/__execution/utility.cuh>
// #include <cuda/experimental/__execution/variant.cuh>
// #include <cuda/experimental/__launch/configuration.cuh>
// #include <cuda/experimental/__launch/launch.cuh>
// #include <cuda/experimental/__stream/stream_ref.cuh>

// #include <nv/target>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <>
struct stream_domain::__apply_t<bulk_t>
{
  template <class _Shape, class _Fn>
  struct __bulk_chunked_fn
  {
    template <class... _Ts>
    _CCCL_TRIVIAL_DEVICE_API auto operator()(_Shape __begin, _Shape __end, _Ts&&... __values) noexcept
    {
      const _Shape __tid = threadIdx.x + blockIdx.x * blockDim.x;

      if (__tid >= __begin && __tid < __end)
      {
        __fn(_Shape(__tid), __values...);
      }
    }

    _Fn __fn_;
  };

  // This function is called from transform_sender when its domain argument is
  // stream_domain and the sender argument is a bulk_t sender. It adapts a bulk sender to
  // the stream domain.
  template <class _Sndr, class _Env>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr, const _Env& __env) const
  {
    // Decompose the bulk sender into its components:
    auto& [__tag, __state, __child] = __sndr;
    auto& [__pol, __shape, __fn]    = __state;

    // It doesn't make sense to use a bulk sender with a non-parallel policy in the stream domain.
    static_assert(_CUDA_VSTD::__is_included_in_v<decltype(__pol), parallel_policy, parallel_unsequenced_policy>);

    using __chunked_fn_t = __bulk_chunked_fn<decltype(__shape), decltype(__fn)>;

    // Adapt the bulk sender to the stream domain:
    return __stream::__adapt(bulk_chunked(
      _CUDA_VSTD::forward_like<_Sndr>(__child), __pol, __shape, __chunked_fn_t{_CUDA_VSTD::forward_like<_Sndr>(__fn)}));
  }
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_BULK
