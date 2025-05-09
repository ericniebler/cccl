//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_GRAPH_CONTEXT
#define __CUDAX__EXECUTION_GRAPH_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/graph/domain.cuh>
#include <cuda/experimental/__execution/graph/visit.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>

#include <cuda_runtime_api.h>

namespace cuda::experimental::execution
{
//////////////////////////////////////////////////////////////////////////////////////////
// graph_context
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_context : private __immovable
{
  struct scheduler;

  struct __tag_t
  {};

  struct __sndr_t
  {
    using sender_concept = sender_t;

    struct __attrs_t
    {
      [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(get_domain_t) noexcept -> graph_domain
      {
        return {};
      }

      [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept
        -> scheduler
      {
        return {__context_};
      }

      [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
      {
        return __context_->__stream_;
      }

      graph_context* __context_;
    };

    template <class _Self>
    [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t(), set_error_t(cudaError_t)>{};
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto get_env() const noexcept
    {
      return __attrs_t{__context_};
    }

    _CCCL_NO_UNIQUE_ADDRESS __tag_t __tag_;
    graph_context* __context_;
  };

  struct scheduler
  {
    using scheduler_concept = scheduler_t;

    [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(get_domain_t) noexcept -> graph_domain
    {
      return {};
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(get_stream_t) noexcept -> stream_ref
    {
      return __context_->__stream_;
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto schedule() const noexcept -> __sndr_t
    {
      return {{}, __context_};
    }

    [[nodiscard]] _CCCL_TRIVIAL_API friend constexpr auto operator==(const scheduler&, const scheduler&) noexcept
      -> bool
    {
      return true;
    }

    [[nodiscard]] _CCCL_TRIVIAL_API friend constexpr auto operator!=(const scheduler&, const scheduler&) noexcept
      -> bool
    {
      return false;
    }

    graph_context* __context_;
  };

  _CCCL_TRIVIAL_API graph_context(stream_ref __stream) noexcept
      : __stream_(__stream)
  {}

  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto get_scheduler() noexcept -> scheduler
  {
    return {this};
  }

  stream_ref __stream_;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_domain::__apply_t<graph_context::__tag_t>
{
  using __result_t = __variant<_CUDA_VSTD::__tuple<set_value_t>, _CUDA_VSTD::__tuple<set_error_t, cudaError_t>>;

  template <class _Context>
  _CCCL_HOST_API auto operator()(_Context& __context, graph_context*) -> __visit_result_t
  {
    __visit_result_t __result{__context.__graph_};
    _CCCL_TRY_CUDA_API(
      cudaGraphAddEmptyNode,
      "Failed to allocate the starting graph node.",
      &__result.__node_.__node_,
      __context.__graph_.__graph_,
      nullptr,
      0);

    // Reserve space for the result variant in the temp storage:
    __result.__result_id_ = __context.__storage_registry_.template __reserve_for<__result_t>();

    // Queue an update that will initialize the result variant in the temp storage:
    __context.__pending_updates_.emplace_back([__result_id = __result.__result_id_](__storage_registry __registry) {
      auto& __result = __registry.__read_at<__result_t>(__result_id);
      __result.__emplace<_CUDA_VSTD::__tuple<set_value_t>>();
    });
    return __result;
  }
};

template <>
inline constexpr auto structured_binding_size<graph_context::__sndr_t> = 2;

} // namespace cuda::experimental::execution

#endif // __CUDAX__EXECUTION_GRAPH_CONTEXT
