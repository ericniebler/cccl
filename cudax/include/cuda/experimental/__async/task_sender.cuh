//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_TASK_SENDER__
#define __CUDAX_ASYNC_TASK_SENDER__

#include <cuda/__cccl_config>

#include "cuda/experimental/__detail/config.cuh"
#include "cuda/std/__internal/namespaces.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/optional>

#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/sender.cuh>
#include <cuda/experimental/__stream/get_stream.cuh>

#include <exception>

namespace cuda::experimental
{
inline ::std::exception_ptr __as_eptr(cudaError_t __status)
{
  return ::std::make_exception_ptr(cuda::cuda_error(__status, "Oopsie"));
}

template <class _Value, class _Rcvr>
struct __task_state
{
  _Rcvr __rcvr_;
  cudaError_t __status_{cudaSuccess};
  _CUDA_VSTD::optional<_Value> __value_{};
};

template <class _Value, class _Rcvr>
struct __thunk_receiver
{
  using receiver_concept = __async::receiver_t;

  void set_value() && noexcept
  {
    if (__state_->__status_ == cudaSuccess)
    {
      __async::set_value(_CUDA_VSTD::move(__state_->__rcvr_), _CUDA_VSTD::move(*__state_->__value_));
    }
    else
    {
      __async::set_error(_CUDA_VSTD::move(__state_->__rcvr_), __as_eptr(__state_->__status_));
    }
  }

  template <class _Error>
  void set_error(_Error error) && noexcept
  {
    __async::set_error(_CUDA_VSTD::move(__state_->__rcvr_), _CUDA_VSTD::move(error));
  }

  void set_stopped() && noexcept
  {
    __async::set_stopped(_CUDA_VSTD::move(__state_->__rcvr_));
  }

  auto get_env() const noexcept -> __async::env_of_t<_Rcvr>
  {
    return __async::get_env(__state_->__rcvr_);
  }

  __task_state<_Value, _Rcvr>* __state_;
};

template <class...>
using __swallow_values = __async::completion_signatures<>;
template <class Error>
using __pass_thru_errors  = __async::completion_signatures<__async::set_error_t(Error)>;
using __pass_thru_stopped = __async::completion_signatures<__async::set_stopped_t()>;

template <class _Task, class _Env>
using __task_completions = __async::transform_completion_signatures_of<
  __async::schedule_result_t<_CUDA_VSTD::__call_result_t<__async::get_scheduler_t, _Env>>,
  _Env,
  __async::completion_signatures<__async::set_value_t(__task_value_of<_Task>),
                                 __async::set_error_t(::std::exception_ptr)>,
  __swallow_values,
  __pass_thru_errors,
  __pass_thru_stopped>;

template <class _Task, class _Rcvr>
struct __task_opstate : __task_state<__task_value_of<_Task>, _Rcvr>
{
  using __env_t        = __async::env_of_t<_Rcvr>;
  using __scheduler_t  = _CUDA_VSTD::__call_result_t<__async::get_scheduler_t, __env_t>;
  using __sch_sndr_t   = __async::schedule_result_t<__scheduler_t>;
  using __thunk_rcvr_t = __thunk_receiver<__task_value_of<_Task>, _Rcvr>;

  using operation_state_concept = __async::operation_state_t;
  using completion_signatures   = __task_completions<_Task, __env_t>;

  static void stream_callback(cudaStream_t, cudaError_t __status, void* __user_data)
  {
    auto* __self      = static_cast<__task_opstate*>(__user_data);
    __self->__status_ = __status;
    // Transfer execution to the current CPU scheduler.
    __async::start(__self->__reschedule_);
  }

  __task_opstate(_Task task, _Rcvr __rcvr)
      : __task_state<__task_value_of<_Task>, _Rcvr>{_CUDA_VSTD::move(__rcvr)}
      , __task_(_CUDA_VSTD::move(task))
      , __reschedule_(__async::connect(
          __async::schedule(__async::get_scheduler(__async::get_env(this->__rcvr_))), __thunk_rcvr_t{this}))
  {}

  __task_opstate(__task_opstate&&) = delete;

  void start() & noexcept
  try
  {
    stream_ref __stream = get_stream(__async::get_env(this->__rcvr_));
    this->__value_.emplace(_CUDA_VSTD::move(__task_).enqueue(__stream));
    cudaError_t __status = cudaStreamAddCallback(__stream.get(), &stream_callback, this, 0);
    if (__status != cudaSuccess)
    {
      __async::set_error(_CUDA_VSTD::move(this->__rcvr_), __as_eptr(__status));
    }
  }
  catch (...)
  {
    __async::set_error(_CUDA_VSTD::move(this->__rcvr_), ::std::current_exception());
  }

private:
  _Task __task_;
  __async::connect_result_t<__sch_sndr_t, __thunk_rcvr_t> __reschedule_;
};

struct _CANNOT_TURN_TASK_INTO_A_SENDER;
struct _ENVIRONMENT_DOES_NOT_HAVE_A_VALUE_FOR_THE_REQUESTED_QUERY;

template <class _Env, class _Query>
struct __missing_query_error
{
  using operation_state_concept = __async::operation_state_t;
  using completion_signatures =
    __async::_ERROR<__async::_WHAT(_CANNOT_TURN_TASK_INTO_A_SENDER),
                    __async::_WHY(_ENVIRONMENT_DOES_NOT_HAVE_A_VALUE_FOR_THE_REQUESTED_QUERY),
                    __async::_WITH_QUERY(_Query),
                    __async::_WITH_ENVIRONMENT(_Env)>;
};

struct _WITH_TASK;
struct _TASK_TYPE_IS_NOT_COPY_CONSTRUCTIBLE;

template <class _Task>
struct __move_only_task_error
{
  using operation_state_concept = __async::operation_state_t;
  using completion_signatures =
    __async::_ERROR<__async::_WHAT(_TASK_TYPE_IS_NOT_COPY_CONSTRUCTIBLE), _WITH_TASK(_Task)>;
};

template <class _Task>
struct __task_sndr
{
  using sender_concept = __async::sender_t;

  _CUDAX_TRIVIAL_HOST_API __task_sndr(_Task __task)
      : __task_(_CUDA_VSTD::move(__task))
  {}

  template <class _Rcvr>
  _CUDAX_TRIVIAL_HOST_API auto connect(_Rcvr __rcvr) &&
  {
    return __connect_(_CUDA_VSTD::move(*this), _CUDA_VSTD::move(__rcvr));
  }

  template <class _Rcvr>
  _CUDAX_TRIVIAL_HOST_API auto connect(_Rcvr __rcvr) const&
  {
    if constexpr (!_CUDA_VSTD::copy_constructible<_Task>)
    {
      return __move_only_task_error<_Task>{};
    }
    else
    {
      return __connect_(*this, _CUDA_VSTD::move(__rcvr));
    }
  }

private:
  template <class _Self, class _Rcvr>
  _CUDAX_TRIVIAL_HOST_API static auto __connect_(_Self&& __self, _Rcvr __rcvr)
  {
    using _Env = __async::env_of_t<_Rcvr>;
    if constexpr (!_CUDA_VSTD::__is_callable_v<__async::get_scheduler_t, _Env>)
    {
      return __missing_query_error<_Env, __async::get_scheduler_t>{};
    }
    else if constexpr (!_CUDA_VSTD::__is_callable_v<get_stream_t, _Env>)
    {
      return __missing_query_error<_Env, get_stream_t>{};
    }
    else
    {
      return __task_opstate<_Task, _Rcvr>{_CUDA_VSTD::forward_like<_Self>(__self.__task_), _CUDA_VSTD::move(__rcvr)};
    }
  }

  _Task __task_;
};

_CCCL_TEMPLATE(class _Task)
_CCCL_REQUIRES(Task<_Task>)
_CUDAX_HOST_API auto as_sender(_Task task) -> __task_sndr<_Task>
{
  return __task_sndr<_Task>{_CUDA_VSTD::move(task)};
}

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_TASK_SENDER__
