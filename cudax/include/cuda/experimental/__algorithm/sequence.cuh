//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ALGORITHM_SEQUENCE
#define __CUDAX_ALGORITHM_SEQUENCE

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

template <class... _Handles>
struct __sequence_handle
{
  friend auto __cudax_unpack_future(__sequence_handle<_Handles...>& __self)
  {
    return cuda::std::apply(
      [](_Handles&... __handles) {
        return (__unpack(__handles), ...); // return the last
      },
      __self.__handles_);
  }

  cuda::std::tuple<_Handles...> __handles_;
};

template <typename... _Tokens>
struct __sequence_action
{
  using __sequence_handle_t = __sequence_handle<__action_result_t<_Tokens>...>;

  __sequence_handle_t __enqueue(stream_ref __stream) &&
  {
    return cuda::std::apply(
      [](auto&... __tokens) {
        return sequence_handle_t{{cuda::std::move(__tokens).__enqueue(__stream)...}};
      },
      __tokens_);
  }

  cuda::std::tuple<_Tokens...> __tokens_;
};

_CCCL_TEMPLATE(typename... _Tokens)
_CCCL_REQUIRES((__is_cudax_action<_Tokens> && ...))
sequence_action<_Tokens...> sequence(_Tokens... __tokens)
{
  return sequence_action<_Tokens...>{{cuda::std::move(__tokens)...}};
}

} // namespace cuda::experimental
#endif // __CUDAX_ALGORITHM_SEQUENCE
