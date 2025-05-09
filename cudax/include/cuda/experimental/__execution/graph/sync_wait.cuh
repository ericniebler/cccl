//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_GRAPH_CONTEXT
#define __CUDAX_ASYNC_DETAIL_GRAPH_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/pod_tuple.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/variant>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/graph/domain.cuh>
#include <cuda/experimental/__execution/graph/visit.cuh>
#include <cuda/experimental/__execution/sync_wait.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__graph/graph.cuh>

#include <cuda_runtime_api.h>

namespace cuda::experimental::execution
{
/////////////////////////////////////////////////////////////////////////////////
// sync_wait: customization for the graph scheduler
template <>
struct graph_domain::__apply_t<sync_wait_t>
{
  template <class _Tag, class... _Values>
  using __value_tuple_t = _CUDA_VSTD::tuple<_Values...>;

  template <class _Values>
  struct __result_visitor
  {
    template <class... _Args>
    _CCCL_HOST_API void operator()(set_value_t, _Args&... __args) const
    {
      __result_.emplace(_CCCL_MOVE(__args)...);
    }

    _CCCL_HOST_API void operator()(set_error_t, cudaError_t __err) const
    {
      __throw_cuda_error(__err, "graph execution failed");
    }

    _CCCL_HOST_API void operator()(set_error_t, _CUDA_VSTD::__ignore_t) const
    {
      __throw_cuda_error(cudaErrorUnknown, "graph execution failed");
    }

    _CCCL_HOST_API void operator()(set_stopped_t) const
    {
      // no-op
    }

    template <class _Tuple>
    _CCCL_HOST_API void operator()(_Tuple& __tuple) const
    {
      _CUDA_VSTD::__apply(*this, __tuple);
    }

    _CUDA_VSTD::optional<_Values>& __result_;
  };

  template <class _Sndr, class... _Env>
  _CCCL_HOST_API auto operator()(_Sndr&& __sndr, _Env&&...) const
  {
    using __completions_t = completion_signatures_of_t<_Sndr, graph_domain::__context_t>;

    if constexpr (!__valid_completion_signatures<__completions_t>)
    {
      return __bad_sync_wait<__completions_t>{};
    }
    else
    {
      // static_assert(sizeof(__sndr) == 0);
      using __sndr_result_t = typename __completions_t::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;

      // transform the sender to a graph:
      auto __stream  = get_stream(get_env(__sndr)); // TODO: not right
      auto __context = graph_domain::__context_t{__stream};
      auto __tmp     = execution::visit(__graph_visitor, static_cast<_Sndr&&>(__sndr), __context);

      auto __registry = __context.__finalize();

      auto __exec = __context.__graph_.instantiate(); // TODO: cache this?
      __exec.launch(__stream);
      __stream.sync();
      // The results of the execution are in the registry:
      auto& __output = __registry.__read_at<__sndr_result_t>(__tmp.__result_id_);

      // Convert the result to an optional<tuple<values...>>:
      using __values_t _CCCL_NODEBUG_ALIAS =
        __value_types<__completions_t, _CUDA_VSTD::tuple, _CUDA_VSTD::__type_self_t>;

      _CUDA_VSTD::optional<__values_t> __result{};
      __output.__visit(__result_visitor<__values_t>{__result}, __output);

      return __result;
    }
  }
};
} // namespace cuda::experimental::execution

#endif // __CUDAX_ASYNC_DETAIL_GRAPH_CONTEXT
