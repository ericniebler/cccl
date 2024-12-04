//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FUTURE_ACCESS
#define __CUDAX_ASYNC_DETAIL_FUTURE_ACCESS

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
#include <cuda/std/__utility/exchange.h>

#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/basic_future.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{

template <class _Task>
_CUDAX_TRIVIAL_HOST_API auto __future_access::__make_future_from_task(_Task __task, stream_ref __stream)
{
  return future<__task_value_of<_Task>>(_CUDA_VSTD::move(__task), __stream);
}

template <class _Value>
_CUDAX_TRIVIAL_HOST_API auto __future_access::__make_future_from_value(_Value __value, stream_ref __stream)
{
  return future<_Value>(_CUDA_VSTD::forward<_Value>(__value), __stream);
}

_CUDAX_TRIVIAL_HOST_API stream_ref __future_access::__exchange_stream(const __future_base& __fut, stream_ref __stream)
{
  return _CUDA_VSTD::exchange(__fut.__stream_, __stream);
}

template <class _Value>
_CUDAX_TRIVIAL_HOST_API auto __future_access::__make_shared_future(future<_Value>&& __fut) -> shared_future<_Value>
{
  return shared_future<_Value>(_CUDA_VSTD::move(__fut));
}

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_FUTURE_ACCESS
