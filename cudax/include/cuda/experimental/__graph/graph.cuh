//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH
#define __CUDAX_GRAPH_GRAPH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/span>

#include <cuda/experimental/__graph/graph_node.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
//! \brief A RAII wrapper for managing a CUDA graph execution object.
//!
//! The `graph_exec` class provides a safe and convenient interface for managing
//! the lifecycle of a `cudaGraphExec_t` object, ensuring proper cleanup and
//! resource management. It supports move semantics, resource release, and
//! launch of the CUDA graph.
//!
//! \note The `graph_exec` object is not directly constructible. One is obtained
//!       by calling the `instantiate()` method on a `graph` object.
//! \sa cuda::experimental::graph
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_exec
{
  //! \brief Move constructor for `graph_exec`.
  //! \param __other The `graph_exec` object to move from.
  //! \note After the move, the source object is left in a valid but unspecified state.
  _CCCL_HOST_API graph_exec(graph_exec&& __other) noexcept
      : __exec_{_CUDA_VSTD::exchange(__other.__exec_, nullptr)}
  {}

  //! \brief Destructor for `graph_exec`.
  //! \details Ensures proper cleanup of the CUDA graph execution object.
  //! \throws None
  _CCCL_HOST_API ~graph_exec()
  {
    reset();
  }

  //! \brief Move assignment operator for `graph_exec`.
  //! \param __other The `graph_exec` object to move from.
  //! \return A reference to the current object.
  //! \note After the move, the source object is left in a valid but unspecified state.
  //! \throws None
  _CCCL_HOST_API auto operator=(graph_exec __other) noexcept -> graph_exec&
  {
    swap(__other);
    return *this;
  }

  //! \brief Swaps the contents of this `graph_exec` with another.
  //! \param __other The `graph_exec` object to swap with.
  //! \throws None
  _CCCL_HOST_API void swap(graph_exec& __other) noexcept
  {
    _CUDA_VSTD::swap(__exec_, __other.__exec_);
  }

  //! \brief Retrieves the underlying CUDA graph execution object.
  //! \return The `cudaGraphExec_t` handle.
  //! \throws None
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto get() const noexcept -> cudaGraphExec_t
  {
    return __exec_;
  }

  //! \brief Releases ownership of the CUDA graph execution object.
  //! \return The `cudaGraphExec_t` handle, leaving this object in a null state.
  //! \throws None
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto release() noexcept -> cudaGraphExec_t
  {
    return _CUDA_VSTD::exchange(__exec_, nullptr);
  }

  //! \brief Resets the `graph_exec` object, destroying the underlying CUDA graph execution object.
  //! \throws `cuda::std::cuda_error` if `cudaGraphExecDestroy` fails.
  _CCCL_HOST_API void reset() noexcept
  {
    if (auto __exec = _CUDA_VSTD::exchange(__exec_, nullptr))
    {
      _CCCL_ASSERT_CUDA_API(cudaGraphExecDestroy, "cudaGraphDestroy failed", __exec);
    }
  }

  //! \brief Launches the CUDA graph execution object on the specified stream.
  //! \param __stream The stream on which to launch the graph.
  //! \throws `cuda::std::cuda_error` if `cudaGraphLaunch` fails.
  _CCCL_HOST_API void launch(stream_ref __stream)
  {
    _CCCL_TRY_CUDA_API(cudaGraphLaunch, "cudaGraphLaunch failed", __exec_, __stream.get());
  }

private:
  friend struct graph;

  _CCCL_HIDE_FROM_ABI graph_exec() = default;

  cudaGraphExec_t __exec_{};
};

//! @brief A wrapper class for managing CUDA graphs.
//!
//! The `graph` class provides a high-level interface for creating, managing, and
//! manipulating CUDA graphs. It ensures proper resource management and simplifies the
//! process of working with CUDA graph APIs.
//!
//! Features:
//! - Supports construction, destruction, and copying of CUDA graphs.
//! - Provides methods for adding nodes and dependencies to the graph.
//! - Allows instantiation of the graph into an executable form.
//! - Ensures proper cleanup of CUDA resources.
//!
//! Usage:
//! - Create an instance of `graph` to represent a CUDA graph.
//! - Use the `add` methods to add nodes and dependencies to the graph.
//! - Instantiate the graph using the `instantiate` method to obtain an executable graph.
//! - Use the `reset` method to release resources when the graph is no longer needed.
//!
//! Thread Safety:
//! - This class is not thread-safe. Concurrent access to the same `graph` object must be
//!   synchronized externally.
//!
//! Exception Safety:
//! - Methods that interact with CUDA APIs may throw ``cuda::std::cuda_error`` if the
//!   underlying CUDA operation fails.
//! - Move operations leave the source object in a valid but unspecified state.
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph
{
  //! \brief Constructs a new, empty CUDA graph.
  //! \throws `cuda::std::cuda_error` if `cudaGraphCreate` fails.
  _CCCL_HOST_API graph()
  {
    _CCCL_TRY_CUDA_API(cudaGraphCreate, "cudaGraphCreate failed", &__graph_, 0);
  }

  //! \brief Constructs an uninitialized CUDA graph.
  //! \throws None
  _CCCL_HOST_API graph(uninit_t) noexcept {}

  //! \brief Move constructor for `graph`.
  //! \param __other The `graph` object to move from.
  //! \note After the move, the source object is left in a valid but unspecified state.
  //! \throws None
  _CCCL_HOST_API graph(graph&& __other) noexcept
      : __graph_{_CUDA_VSTD::exchange(__other.__graph_, nullptr)}
  {}

  //! \brief Copy constructor for `graph`.
  //! \param __other The `graph` object to copy from.
  //! \throws `cuda::std::cuda_error` if `cudaGraphClone` fails.
  _CCCL_HOST_API graph(const graph& __other)
  {
    if (__other.__graph_)
    {
      _CCCL_TRY_CUDA_API(cudaGraphClone, "cudaGraphClone failed", &__graph_, __other.__graph_);
    }
  }

  //! \brief Destructor for `graph`.
  //! \details Ensures proper cleanup of the CUDA graph object.
  //! \throws None
  _CCCL_HOST_API ~graph()
  {
    reset();
  }

  //! \brief Copy/move assignment operator for `graph`.
  //! \param __other The `graph` object to move from.
  //! \return A reference to the current object.
  //! \note After the move, the source object is left in a valid but unspecified state.
  //! \throws None
  _CCCL_HOST_API auto operator=(graph __other) noexcept -> graph&
  {
    swap(__other);
    return *this;
  }

  //! \brief Swaps the contents of this `graph` with another.
  //! \param __other The `graph` object to swap with.
  //! \throws None
  _CCCL_HOST_API void swap(graph& __other) noexcept
  {
    _CUDA_VSTD::swap(__graph_, __other.__graph_);
  }

  //! \brief Retrieves the underlying CUDA graph object.
  //! \return The `cudaGraph_t` handle.
  //! \throws None
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto get() const noexcept -> cudaGraph_t
  {
    return __graph_;
  }

  //! \brief Releases ownership of the CUDA graph object.
  //! \return The `cudaGraph_t` handle, leaving this object in a null state.
  //! \throws None
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto release() noexcept -> cudaGraph_t
  {
    return _CUDA_VSTD::exchange(__graph_, nullptr);
  }

  //! \brief Resets the `graph` object, destroying the underlying CUDA graph object.
  //! \throws `cuda::std::cuda_error` if `cudaGraphDestroy` fails.
  _CCCL_HOST_API void reset() noexcept
  {
    if (auto __graph = _CUDA_VSTD::exchange(__graph_, nullptr))
    {
      _CCCL_ASSERT_CUDA_API(cudaGraphDestroy, "cudaGraphDestroy failed", __graph);
    }
  }

  //! \brief Adds a new root node to the graph.
  //! \tparam _Node The type of the node to add.
  //! \param __node The node to add to the graph.
  //! \return A `graph_node` representing the added node.
  //! \throws `cuda::std::cuda_error` if adding the node fails.
  template <class _Node>
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto add(_Node __node) -> graph_node
  {
    return __node.__add_to_graph(__graph_);
  }

  //! \brief Adds a new node with dependencies to the graph.
  //! \tparam _Node The type of the node to add.
  //! \tparam _Np The number of dependencies.
  //! \param __node The node to add to the graph.
  //! \param __deps The dependencies for the new node.
  //! \return A `graph_node` representing the added node.
  //! \throws `cuda::std::cuda_error` if adding the node or dependencies fails.
  template <class _Node, size_t _Np>
  _CCCL_HOST_API auto add(_Node __node, _CUDA_VSTD::array<cudaGraphNode_t, _Np> __deps) -> graph_node
  {
    auto __new_node = add(__node);
    cudaGraphNode_t __src_arr[_Np];
    _CUDA_VSTD::fill(__src_arr, __src_arr + _Np, __new_node.__node_);
    _CCCL_TRY_CUDA_API(
      cudaGraphAddDependencies,
      "cudaGraphAddDependencies failed",
      __graph_,
      __src_arr, // dependencies
      __deps.__nodes_,
      _Np); // number of dependencies
    return __new_node;
  }

  //! \brief Instantiates the CUDA graph into a `graph_exec` object.
  //! \return A `graph_exec` object representing the instantiated graph.
  //! \throws `cuda::std::cuda_error` if `cudaGraphInstantiate` fails.
  _CCCL_HOST_API auto instantiate() -> graph_exec
  {
    _CCCL_ASSERT(__graph_ != nullptr, "cannot instantiate a NULL graph");
    graph_exec __exec;
    _CCCL_TRY_CUDA_API(
      cudaGraphInstantiate,
      "cudaGraphInstantiate failed",
      &__exec.__exec_, // output
      __graph_, // graph to instantiate
      0); // flags
    return __exec;
  }

private:
  //! \brief Adds this graph as a child graph to the parent graph.
  //! \param __parent The parent graph to which this graph will be added.
  //! \return A `graph_node` representing the added child graph.
  //! \throws `cuda::std::cuda_error` if `cudaGraphAddChildGraphNode` fails.
  [[nodiscard]] _CCCL_HOST_API auto __add_to_graph(cudaGraph_t __parent) -> graph_node
  {
    graph_node __child;
    __child.__graph_ = __graph_;
    _CCCL_ASSERT_CUDA_API(
      cudaGraphAddChildGraphNode,
      "cudaGraphAddChildGraphNode failed",
      &__child.__node_, // output
      __parent, // graph to which we are adding the child graph
      nullptr, // dependencies
      0, // number of dependencies
      __graph_); // the child graph to add
    return __child;
  }

  cudaGraph_t __graph_{};
};
} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_GRAPH_GRAPH
