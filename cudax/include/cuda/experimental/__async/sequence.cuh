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
#include <cuda/experimental/__async/get_unsynchronized.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>

namespace cuda::experimental
{

template <class... _Values>
struct __sequence_value
{
  friend decltype(auto) cuda_get_unsynchronized(__sequence_value& __self)
  {
    constexpr size_t __back_index = sizeof...(_Values) - 1;
    return get_unsynchronized(_CUDA_VSTD::get<__back_index>(__self.__values_));
  }

  _CUDA_VSTD::tuple<_Values...> __values_;
};

template <typename... _Tasks>
struct __sequence_task
{
  using __sequence_value_t = __sequence_value<__task_value_of<_Tasks>...>;

  __sequence_value_t enqueue(stream_ref __stream) &&
  {
    return _CUDA_VSTD::apply(
      [__stream](auto&... __tasks) {
        return __sequence_value_t{{_CUDA_VSTD::move(__tasks).enqueue(__stream)...}};
      },
      __tasks_);
  }

  _CUDA_VSTD::tuple<_Tasks...> __tasks_;
};

struct __sequence_t
{
  _CCCL_TEMPLATE(typename... _Tasks)
  _CCCL_REQUIRES((Task<_Tasks> && ...))
  auto operator()(_Tasks... __tasks) const -> __sequence_task<_Tasks...>
  {
    static_assert(sizeof...(_Tasks) > 0, "sequence must have at least one value");
    return __sequence_task<_Tasks...>{{_CUDA_VSTD::move(__tasks)...}};
  }
};

inline constexpr __sequence_t sequence{};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_SEQUENCE__
