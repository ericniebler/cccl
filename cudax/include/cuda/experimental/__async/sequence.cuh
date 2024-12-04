//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_SEQUENCE__
#define __CUDAX_ASYNC_SEQUENCE__

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <cuda/experimental/__async/future.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{

template <class... _Values>
struct __sequence_value
{
  friend auto __cudax_unpack_future(__sequence_value& __self)
  {
    return _CUDA_VSTD::apply(
      [](_Values&... __values) {
        return (__unpack(__values), ...); // return the last
      },
      __self.__values_);
  }

  _CUDA_VSTD::tuple<_Values...> __values_;
};

template <typename... _Actions>
struct __sequence_action
{
  using __sequence_value_t = __sequence_value<__action_result_t<_Actions>...>;

  __sequence_value_t __enqueue(stream_ref __stream) &&
  {
    return _CUDA_VSTD::apply(
      [__stream](auto&... __actions) {
        return __sequence_value_t{{_CUDA_VSTD::move(__actions).__enqueue(__stream)...}};
      },
      __actions_);
  }

  _CUDA_VSTD::tuple<_Actions...> __actions_;
};

_CCCL_TEMPLATE(typename... _Actions)
_CCCL_REQUIRES((__is_cudax_action<_Actions> && ...))
__sequence_action<_Actions...> sequence(_Actions... __actions)
{
  return __sequence_action<_Actions...>{{_CUDA_VSTD::move(__actions)...}};
}

} // namespace cuda::experimental
#endif // __CUDAX_ASYNC_SEQUENCE__
