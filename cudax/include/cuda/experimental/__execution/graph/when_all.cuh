//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_GRAPH_THEN
#define __CUDAX__EXECUTION_GRAPH_THEN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__execution/graph/domain.cuh>
#include <cuda/experimental/__execution/graph/visit.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/when_all.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_domain::__apply_t<when_all_t>
{
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __result_visitor_t
  {
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __unpack_t
    {
      template <class _Fn, class _Variant, class _Tag, class... _Args>
      _CCCL_DEVICE_API void operator()(_Fn& __fn, _Variant& __result, _Tag, _Args&... __args) const
      {
        if constexpr (_Tag() == set_value)
        {
          using __result_t = _CUDA_VSTD::invoke_result_t<_Fn, _Args...>;
          using __tuple_t  = _CUDA_VSTD::__decayed_tuple<set_value_t, __result_t>;
          __result.template __emplace<__tuple_t>(
            _Tag(), _CUDA_VSTD::invoke(static_cast<_Fn&&>(__fn), static_cast<_Args&&>(__args)...));
        }
        else
        {
          using __tuple_t = _CUDA_VSTD::__decayed_tuple<_Tag, _Args...>;
          __result.template __emplace<__tuple_t>(_Tag(), static_cast<_Args&&>(__args)...);
        }
      }
    };

    template <class _Fn, class _Variant, class _Tuple>
    _CCCL_DEVICE_API void operator()(_Fn& __fn, _Variant& __result, _Tuple& __tuple) const
    {
      _CUDA_VSTD::__apply(__unpack_t{}, __tuple, __fn, __result);
    }
  };

  template <class _Context, class _Fn, class _Sndr>
  _CCCL_HOST_API auto operator()(_Context& __context, _Fn&& __fn, _Sndr&& __sndr) const -> __visit_result_t
  {
    // TODO: check for type errors:
    // Compute the variant needed to hold the predecessor sender's result:
    using __pred_completions_t = completion_signatures_of_t<_Sndr, _Context>;
    using __pred_result_t =
      typename __pred_completions_t::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;

    // Compute the variant needed to hold the then sender's result:
    using __self_t             = decltype(then(declval<_Sndr>(), declval<_Fn>()));
    using __self_completions_t = completion_signatures_of_t<__self_t, _Context>;
    using __self_result_t =
      typename __self_completions_t::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;

    // transform the child sender to a graph node:
    auto __pred_result = execution::visit(__graph_visitor, static_cast<_Sndr&&>(__sndr), __context);

    // Reserve temporary storage for the result of the "then" sender:
    const auto __self_id = __context.__storage_registry_.template __reserve_for<__self_result_t>();

    // Create a sender for a new kernel node in the graph:
    auto __tmp = launch(
      __default_config,
      // The first argument must be an empty __storage_registry object so that the graph visitor
      // can update it once the temporary storage is allocated:
      []
      _CCCL_DEVICE(_CUDA_VSTD::__ignore_t, __storage_registry __storage, size_t __pred_id, size_t __self_id, _Fn __fn)
      -> void {
        // This is the variant that holds the result of the predecessor sender:
        auto& __pred_result = __storage.__read_at<__pred_result_t>(__pred_id);
        // This is the variant that holds the result of the then sender:
        auto& __self_result = __storage.__write_at<__self_result_t>(__self_id);
        // This reads the result of the predecessor sender and invokes __fn with the value
        // result. Other result kinds are forwarded unchanged.
        __pred_result.__visit(__result_visitor_t{}, __pred_result, __fn, __self_result);
      },
      __storage_registry{},
      __pred_result.__result_id_,
      __self_id,
      _CCCL_MOVE(__fn));

    // Turn the launch sender into a graph node:
    auto __self_result = execution::visit(__graph_visitor, _CCCL_MOVE(__tmp), __context);
    _CCCL_ASSERT(__self_result.__result_id_ == static_cast<size_t>(-1),
                 "kernel nodes should not reserve temporary storage");
    __self_result.__result_id_ = __self_id;

    // The "then" kernel node depends on the predecessor node:
    __self_result.__node_.depends_on(__pred_result.__node_);

    // Add the kernel node to the list of nodes with pending updates:
    __context.__pending_updates_.emplace_back([__node = __self_result.__node_.__node_](__storage_registry __registry) {
      // Read the parameters of the kernel node
      cudaKernelNodeParams __params{};
      _CCCL_TRY_CUDA_API(cudaGraphKernelNodeGetParams, "failed to get kernel graph node parameters", __node, &__params);
      // The first two arguments are __result_id_t objects that contain tokens from the
      // storage registry. We must replace the tokens with the actual pointers to the
      // reserved memory.
      auto& __tmp = *static_cast<__storage_registry*>(__params.kernelParams[__kernel_args_offset]);
      __tmp       = __registry;

      // Update the kernel node parameters with the new values
      _CCCL_TRY_CUDA_API(cudaGraphKernelNodeSetParams, "failed to set kernel graph node parameters", __node, &__params);
    });
    return __self_result;
  }
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif //__CUDAX___EXECUTION_GRAPH_THEN
