//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_GRAPH_VISIT
#define __CUDAX__EXECUTION_GRAPH_VISIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/graph/domain.cuh>
#include <cuda/experimental/__execution/graph/storage_registry.cuh>
#include <cuda/experimental/__graph/graph_builder.cuh>
#include <cuda/experimental/__graph/graph_node_ref.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <functional>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// This is the offset into the cudaKernelNodeParams.kernelParams array where the arguments
// to the __device__ fn passed to cudax::launch are stored. The first two elements are the
// config and the fn itself.
inline constexpr size_t __kernel_args_offset = 2;

// stored in managed memory so that all nodes can access it
struct graph_domain::__state_t
{
  __storage_registry __storage_registry_;
  // Other state information can be added here as needed
};

// passed by reference to the visitaion function
struct graph_domain::__context_t
{
  using __node_update_fn = ::std::function<void(__storage_registry)>;

  _CCCL_HOST_API explicit __context_t(stream_ref __stream) noexcept
      : __storage_registry_{__stream}
      , __state_id_{__storage_registry_.__reserve_for<__state_t>()} // TODO: always zero, optimize away
  {}

  _CCCL_HOST_API auto __finalize() -> __storage_registry
  {
    // Allocate the managed memory for all the temporary storage
    auto __registry = __storage_registry_.__finalize();
    __registry.__write_at<__state_t>(__state_id_, __registry);

    // Tell each kernel node how to look up their reserved storage:
    for (auto& __update_fn : __pending_updates_)
    {
      __update_fn(__registry);
    }
    __pending_updates_.clear();

    return __registry;
  }

  __storage_registry_context __storage_registry_;
  size_t __state_id_{};
  graph_builder __graph_{};
  ::std::vector<__node_update_fn> __pending_updates_{};
};

// returned from the visitation function. it contains the newly created node and a token
// that can be used to retrieve the result of the node.
struct graph_domain::__visit_result_t
{
  _CCCL_HOST_API explicit __visit_result_t(graph_builder& __graph) noexcept
  {
    __node_.__graph_ = __graph.__graph_;
  }

  graph_node_ref __node_{};
  size_t __result_id_{static_cast<size_t>(-1)};
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_GRAPH_VISIT
