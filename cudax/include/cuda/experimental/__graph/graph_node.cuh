//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH_NODE
#define __CUDAX_GRAPH_GRAPH_NODE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/array>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph;
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_node;
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_node_ref;

//! \enum graph_node_type
//!
//! \brief Represents the types of nodes that can exist in a CUDA graph.
//!
//! This enumeration defines various node types that can be used in CUDA graphs
//! to represent different operations or functionalities.
//!
//! \var graph_node_type::kernel
//! Represents a kernel execution node.
//!
//! \var graph_node_type::memcpy
//! Represents a memory copy operation node.
//!
//! \var graph_node_type::memset
//! Represents a memory set operation node.
//!
//! \var graph_node_type::host
//! Represents a host function execution node.
//!
//! \var graph_node_type::graph
//! Represents a nested graph node.
//!
//! \var graph_node_type::empty
//! Represents an empty node with no operation.
//!
//! \var graph_node_type::wait_event
//! Represents a node that waits for an event.
//!
//! \var graph_node_type::event_record
//! Represents a node that records an event.
//!
//! \var graph_node_type::semaphore_signal
//! Represents a node that signals an external semaphore.
//!
//! \var graph_node_type::semaphore_wait
//! Represents a node that waits on an external semaphore.
//!
//! \var graph_node_type::malloc
//! Represents a node that performs memory allocation.
//!
//! \var graph_node_type::free
//! Represents a node that performs memory deallocation.
//!
//! \var graph_node_type::batch_memop
//! Represents a node that performs a batch memory operation.
//!
//! \var graph_node_type::conditional
//! Represents a conditional execution node.
enum class graph_node_type : int
{
  kernel           = cudaGraphNodeTypeKernel,
  memcpy           = cudaGraphNodeTypeMemcpy,
  memset           = cudaGraphNodeTypeMemset,
  host             = cudaGraphNodeTypeHost,
  graph            = cudaGraphNodeTypeGraph,
  empty            = cudaGraphNodeTypeEmpty,
  wait_event       = cudaGraphNodeTypeWaitEvent,
  event_record     = cudaGraphNodeTypeEventRecord,
  semaphore_signal = cudaGraphNodeTypeExtSemaphoreSignal,
  semaphore_wait   = cudaGraphNodeTypeExtSemaphoreWait,
  malloc           = cudaGraphNodeTypeMemAlloc,
  free             = cudaGraphNodeTypeMemFree,
  batch_memop      = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
  conditional      = cudaGraphNodeTypeConditional
};

//! \brief Builds a tuple of graph nodes that represent dependencies. It is for use as a
//!        parameter to the `graph::add` function.
//!
//! \tparam _Nodes Variadic template parameter representing the types of the graph nodes.
//!         Each type must be either `graph_node` or `graph_node_ref`.
//! \param __nodes The graph nodes to add as dependencies to a new node.
//! \return A object of type `cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>`
//!         containing the references to the provided graph nodes.
//!
//! \note A static assertion ensures that all provided arguments are convertible to
//!       `graph_node_ref`. If this condition is not met, a compilation error will occur.
template <class... _Nodes>
_CCCL_TRIVIAL_HOST_API constexpr auto depends_on(_Nodes&&... __nodes) noexcept
  -> _CUDA_VSTD::array<cudaGraphNode_t, sizeof...(_Nodes)>
{
  static_assert((_CUDA_VSTD::is_base_of_v<graph_node_ref, _CUDA_VSTD::remove_reference_t<_Nodes>> && ...),
                "depends_on() requires graph_node arguments");
  return _CUDA_VSTD::array<cudaGraphNode_t, sizeof...(_Nodes)>{{__nodes.get()...}};
}

//! \brief A reference wrapper for a CUDA graph node.
//! This structure provides an interface to manage and interact with a CUDA graph node
//! within a CUDA graph. It includes functionality for swapping, retrieving node information,
//! and managing dependencies between nodes.
struct graph_node_ref
{
  //! \brief Default constructor.
  _CCCL_HIDE_FROM_ABI graph_node_ref() = default;

  //! \brief Constructs a graph_node_ref with a given CUDA graph node and graph.
  //! \param __node The CUDA graph node.
  //! \param __graph The CUDA graph containing the node.
  _CCCL_TRIVIAL_HOST_API explicit graph_node_ref(cudaGraphNode_t __node, cudaGraph_t __graph) noexcept
      : __graph_{__graph}
      , __node_{__node}
  {
    _CCCL_ASSERT(!__node_ == !__graph_, "graph_node_ref: one of __node or __graph is null and the other is not");
  }

  //! \brief Swaps the contents of this graph_node_ref with another.
  //! \param __other The other graph_node_ref to swap with.
  _CCCL_HOST_API void swap(graph_node_ref& __other) noexcept
  {
    _CUDA_VSTD::swap(__graph_, __other.__graph_);
    _CUDA_VSTD::swap(__node_, __other.__node_);
  }

  //! \brief Swaps the contents of two graph_node_ref objects.
  //! \param __left The first graph_node_ref.
  //! \param __right The second graph_node_ref.
  _CCCL_HOST_API friend void swap(graph_node_ref& __left, graph_node_ref& __right) noexcept
  {
    __left.swap(__right);
  }

  //! \brief Retrieves the underlying CUDA graph node.
  //! \return The CUDA graph node.
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto get() const noexcept -> cudaGraphNode_t
  {
    return __node_;
  }

  //! \brief Retrieves the type of the CUDA graph node.
  //! \return The type of the graph node as a graph_node_type.
  //! \throws If the CUDA API call to retrieve the node type fails.
  [[nodiscard]] _CCCL_HOST_API auto type() const -> graph_node_type
  {
    cudaGraphNodeType __type;
    _CCCL_ASSERT_CUDA_API(cudaGraphNodeGetType, "cudaGraphNodeGetType failed", __node_, &__type);
    return static_cast<graph_node_type>(__type);
  }

  //! \brief Establishes dependencies between this node and other nodes.
  //! This function sets up dependencies such that this node depends on the provided nodes.
  //! \tparam _Nodes Variadic template parameter for the types of the dependent nodes.
  //! \param __nodes The nodes that this node depends on.
  //! \throws If the CUDA API call to add dependencies fails.
  template <class... _Nodes>
  _CCCL_HOST_API void depends_on(_Nodes&&... __nodes)
  {
    auto __dependencies      = experimental::depends_on(__nodes...);
    cudaGraphNode_t __self[] = {(true ? __node_ : __nodes.get())...};
    _CCCL_ASSERT_CUDA_API(
      cudaGraphAddDependencies,
      "cudaGraphAddDependencies failed",
      __graph_,
      __dependencies.data(),
      __self,
      sizeof...(__nodes));
  }

private:
  friend struct graph;
  friend struct graph_node;

  cudaGraph_t __graph_{}; ///< The CUDA graph containing the node.
  cudaGraphNode_t __node_{}; ///< The CUDA graph node.
};

//! @brief Represents a node in a CUDA graph, providing RAII-style management of the
//!        underlying CUDA graph node.
//!
//! The `graph_node` class is a wrapper around a CUDA graph node (`cudaGraphNode_t`) and
//! its associated graph (`cudaGraph_t`). It ensures proper resource management by
//! releasing the CUDA graph node when the object is destroyed or reset.
//!
//! This class inherits from `graph_node_ref` and extends its functionality with additional
//! resource management features.
//!
//! ## Key Features:
//! - Default constructor for creating an empty `graph_node`.
//! - Move constructor and move assignment operator for transferring ownership of a CUDA
//!   graph node.
//! - Destructor that automatically releases the CUDA graph node if it is still valid.
//! - `release()` method to release ownership of the CUDA graph node without destroying it.
//! - `reset()` method to explicitly destroy the CUDA graph node.
//! - Static factory method `from_native_handle()` to create a `graph_node` from a native
//!   CUDA graph node handle.
//!
//! ## Usage:
//! This class is designed to be used in CUDA graph-based programming to manage graph nodes
//! safely and efficiently. It ensures that CUDA graph nodes are properly destroyed when
//! they are no longer needed, preventing resource leaks.
//!
//! ## Thread Safety:
//! This class is not thread-safe. Proper synchronization is required if used in a
//! multi-threaded environment.
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_node : graph_node_ref
{
  _CCCL_HIDE_FROM_ABI graph_node() = default;

  //! @brief Move constructor for the `graph_node` class.
  //! @param __other The `graph_node` instance to move from. After the move,
  //!                `__other` will be left in a valid but unspecified state.
  //! @throws None
  _CCCL_HOST_API graph_node(graph_node&& __other) noexcept
      : graph_node_ref{_CUDA_VSTD::exchange(__other.__node_, {}), __other.__graph_}
  {}

  //! @brief Destructor for the `graph_node` class.
  //!
  //! This destructor calls the `reset()` method to destroy the wrapped `cudaGraphNode_t`,
  //! if it is not null.
  _CCCL_HOST_API ~graph_node()
  {
    reset();
  }

  //! @brief Assignment operator for the `graph_node` class.
  //!
  //! This operator assigns the contents of another `graph_node` instance to the current
  //! instance by swapping their contents.
  //!
  //! @param __other The `graph_node` instance to assign from.
  //! @return A reference to the current `graph_node` instance after assignment.
  _CCCL_HOST_API auto operator=(graph_node __other) noexcept -> graph_node&
  {
    swap(__other);
    return *this;
  }

  //! @brief Releases ownership of the current CUDA graph node.
  //!
  //! This function releases the ownership of the CUDA graph node managed by this object
  //! and returns the underlying `cudaGraphNode_t`. After calling this function, the
  //! internal node pointer is set to `nullptr`, and the caller assumes responsibility
  //! for managing the returned node.
  //!
  //! @return cudaGraphNode_t The CUDA graph node that was previously managed by this object.
  //!         If the internal node pointer was already `nullptr`, this function returns `nullptr`.
  //!
  _CCCL_TRIVIAL_HOST_API auto release() noexcept -> cudaGraphNode_t
  {
    return _CUDA_VSTD::exchange(__node_, nullptr);
  }

  //! @brief Resets the graph node by destroying the current CUDA graph node, if it exists.
  //!
  //! This function ensures that the current CUDA graph node is destroyed and the internal
  //! pointer is set to `nullptr`. It uses `cudaGraphDestroyNode` to release the resources
  //! associated with the node. If the destruction fails, an assertion is triggered.
  _CCCL_HOST_API void reset() noexcept
  {
    if (auto __old = _CUDA_VSTD::exchange(__node_, nullptr))
    {
      _CCCL_ASSERT_CUDA_API(cudaGraphDestroyNode, "cudaGraphDestroy failed", __old);
    }
  }

  //! @brief Creates a `graph_node` object from a native CUDA graph node handle and graph
  //!        handle.
  //!
  //! @param __node The native CUDA graph node handle (`cudaGraphNode_t`).
  //! @param __graph The native CUDA graph handle (`cudaGraph_t`) to which the node belongs.
  //! @return A `graph_node` object representing the specified CUDA graph node.
  //! @pre The `__node` and `__graph` parameters must both be non-null, or both must be
  //!      null. If one is null and the other is not, an assertion will be triggered.
  _CCCL_HOST_API static auto from_native_handle(cudaGraphNode_t __node, cudaGraph_t __graph) noexcept -> graph_node
  {
    return graph_node{__node, __graph};
  }

private:
  friend struct graph;

  _CCCL_HOST_API explicit graph_node(cudaGraphNode_t __node, cudaGraph_t __graph) noexcept
      : graph_node_ref{__node, __graph}
  {}
};

} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_GRAPH_GRAPH_NODE
