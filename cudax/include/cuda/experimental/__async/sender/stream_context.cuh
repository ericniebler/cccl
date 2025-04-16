//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_STREAM_CONTEXT
#define __CUDAX_ASYNC_DETAIL_STREAM_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/sender/domain.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/stream.cuh>

#include <new> // IWYU pragma: keep

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
//////////////////////////////////////////////////////////////////////////////////////////
// stream domain
struct stream_domain : default_domain
{};

inline stream_ref __gc_stream() noexcept
{
  static stream str;
  return str;
}

namespace
{
template <class _Fn, class... _Args>
__global__ void __stream_invoke(_Fn __fn, _Args... __args)
{
  static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...);
}

template <class _Fn, class... _Args>
__global__ void __stream_invoke_r(__call_result_t<_Fn, _Args...>* __return, _Fn __fn, _Args... __args)
{
  using _Return = __call_result_t<_Fn, _Args...>;
  ::new (static_cast<void*>(__return)) _Return(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...));
}
} // namespace

//////////////////////////////////////////////////////////////////////////////////////////
// stream_context
struct _CCCL_TYPE_VISIBILITY_DEFAULT stream_context : private __immovable
{
  stream_context() noexcept = default;

  _CUDAX_HOST_API void sync() noexcept
  {
    __stream_.sync();
  }

  _CUDAX_HOST_API auto get_scheduler() noexcept;

private:
  struct __scheduler;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __complete_fn
  {
    template <class _Tag, class _Rcvr, class... _Args>
    _CUDAX_HOST_API void operator()(_Tag, _Rcvr& __rcvr, _Args&&... __args) const noexcept
    {
      _Tag{}(static_cast<_Rcvr&&>(__rcvr), static_cast<_Args&&>(__args)...);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // environment of the stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    _CUDAX_HOST_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept
    {
      return __scheduler{__stream_ref_};
    }

    _CUDAX_HOST_API constexpr auto query(get_stream_t) const noexcept
    {
      return __stream_ref_;
    }

    _CUDAX_TRIVIAL_HOST_API static constexpr auto query(get_domain_t) noexcept
    {
      return stream_domain{};
    }

    stream_ref __stream_ref_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's operation state
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : private __immovable
  {
    using operation_state_concept = operation_state_t;

    _CUDAX_HOST_API void start() & noexcept
    {
      __stream_invoke<<<1, 1, 0, __stream_ref_.get()>>>(__complete_fn{}, set_value, __rcvr_);
      if (auto status = cudaGetLastError(); status != cudaSuccess)
      {
        set_error(static_cast<_Rcvr&&>(__rcvr_), status);
      }
    }

    _Rcvr __rcvr_;
    stream_ref __stream_ref_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class _Self>
    _CUDAX_HOST_API static constexpr auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t(), set_error_t(cudaError_t)>{};
    }

    _CUDAX_HOST_API constexpr auto get_env() const noexcept -> __env_t const&
    {
      return __env_;
    }

    template <class _Rcvr>
    _CUDAX_HOST_API auto connect(_Rcvr __rcvr) const noexcept
    {
      return __opstate_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __env_.__stream_ref_};
    }

    __env_t __env_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __scheduler
  {
    using scheduler_concept = scheduler_t;

    _CUDAX_HOST_API bool operator==(const __scheduler& __other) const noexcept
    {
      return __stream_ref_ == __other.__stream_ref_;
    }

    _CUDAX_HOST_API static constexpr auto query(get_forward_progress_guarantee_t) noexcept
    {
      return forward_progress_guarantee::weakly_parallel;
    }

    _CUDAX_HOST_API static constexpr auto query(get_domain_t) noexcept
    {
      return stream_domain{};
    }

    _CUDAX_HOST_API auto schedule() const noexcept
    {
      return __sndr_t{{__stream_ref_}};
    }

    stream_ref __stream_ref_;
  };

  stream __stream_{};
};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_STREAM_CONTEXT
