//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_SEQUENCE
#define __CUDAX_EXECUTION_STREAM_SEQUENCE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/diagnostics.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/sequence.cuh>
#include <cuda/experimental/__execution/stream/adaptor.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/stream/scheduler.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CANNOT_DISPATCH_THE_SEQUENCE_ALGORITHM_TO_THE_STREAM_SCHEDULER;
struct _BECAUSE_THERE_IS_NO_STREAM_SCHEDULER_IN_THE_ENVIRONMENT;
struct _ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STREAM_SCHEDULER_BEFORE_THE_SEQUENCE_ALGORITHM;

namespace __stream
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t
{
  template <class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  // This function is called when the connect CPO is passed a sequence sender that
  // completes on the stream domain.
  template <class _Sndr, class _Env>
  _CCCL_API auto operator()(set_value_t, _Sndr&& __sndr, const _Env& __env) const
  {
    auto& [__tag, __ign, __child1, __child2] = __sndr;
    using __child1_t                         = decltype(__child1);
    using __child2_t                         = decltype(__child2);
    using __domain1_t = __completion_domain_of_t<set_value_t, __child1_t, __fwd_env_t<const _Env&>>;

    // We only need to transform the sequence sender when the first child sender
    // completes on the stream domain. The second child sender needs to be started
    // on the host.
    if constexpr (__same_as<__domain1_t, stream_domain>)
    {
      if constexpr (__completes_on<__child1_t, stream_scheduler, __fwd_env_t<const _Env&>>)
      {
        auto __sch     = get_completion_scheduler<set_value_t>(execution::get_env(__child1), __fwd_env(__env));
        using __sndr_t = sequence_t::__sndr_t<__child1_t, __child2_t>;
        return __sndr_t(__sch, static_cast<__child1_t&&>(__child1), static_cast<__child2_t&&>(__child2));
      }
      else
      {
        return __not_a_sender<
          _WHAT(_CANNOT_DISPATCH_THE_SEQUENCE_ALGORITHM_TO_THE_STREAM_SCHEDULER),
          _WHY(_BECAUSE_THERE_IS_NO_STREAM_SCHEDULER_IN_THE_ENVIRONMENT),
          _WHERE(_IN_ALGORITHM, execution::sequence_t),
          _TO_FIX_THIS_ERROR(_ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STREAM_SCHEDULER_BEFORE_THE_SEQUENCE_ALGORITHM),
          _WITH_SENDER(__child1_t),
          _WITH_ENVIRONMENT(_Env)>();
      }
    }
    else
    {
      return static_cast<_Sndr&&>(__sndr);
    }
  }

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env2_t : prop<get_scheduler_t, stream_scheduler>
  {
    _CCCL_API constexpr explicit __env2_t(stream_scheduler __sch) noexcept
        : __env2_t::prop{{}, static_cast<stream_scheduler&&>(__sch)}
    {}
  };

  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    __rcvr_with_env_t<_Rcvr, __env2_t> __rcvr_env_;
    bool __success_ = false;
  };

  // This receiver is used to connect to the first sender in the sequence. When it
  // receives set_value, it sets __state_.__success_ to true, indicating that
  // the second sender should be started. For the other completions, it forwards
  // them to the original receiver inline.
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr1_t
  {
    using receiver_concept = receiver_t;

    _CCCL_API constexpr void set_value() noexcept
    {
      __state_.__success_ = true;
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_.__rcvr_env_.__base()), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_.__rcvr_env_.__base()));
    }

    [[nodiscard]]
    _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_.__rcvr_env_.__base()));
    }

    __state_t<_Rcvr>& __state_;
  };

  template <class _Rcvr>
  _CCCL_HOST_DEVICE __rcvr1_t(__state_t<_Rcvr>&) -> __rcvr1_t<_Rcvr>;

  // This receiver is used to connect to the second sender in the sequence. It uses
  // the receiver environment type __env2_t, which contains the stream scheduler.
  // It simply forwards all completions to the original receiver.
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr2_t : __rcvr_ref<__rcvr_with_env_t<_Rcvr, __env2_t>>
  {
    using __base_rcvr_t = __rcvr_with_env_t<_Rcvr, __env2_t>;
    _CCCL_API constexpr explicit __rcvr2_t(__base_rcvr_t& __rcvr) noexcept
        : __rcvr2_t::__rcvr_ref(__rcvr)
    {}
  };

  template <class _Rcvr, class _CvSndr1, class _CvSndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _CCCL_API constexpr __opstate_t(_Rcvr&& __rcvr, _CvSndr1&& __child1, _CvSndr2&& __child2)
        : __state_(__opstate_t::__mk_state(__child1, __rcvr))
        , __op1_(execution::connect(static_cast<_CvSndr1&&>(__child1), __rcvr1_t{__state_}))
        , __op2_(execution::connect(static_cast<_CvSndr2&&>(__child2), __rcvr2_t{__state_.__rcvr_env_}))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      NV_IF_TARGET(NV_IS_HOST, //
                   (__host_start();),
                   (__device_start();));
    }

  private:
    _CCCL_HOST_API void __host_start() noexcept
    {
      _CCCL_TRY
      {
        execution::start(__op1_);

        // Synchronize the stream before starting the second operation.
        stream_ref __stream = get_stream(get_scheduler(get_env(__state_.__rcvr_env_)));
        __stream.sync(); // potentially throwing

        // If __state_.__success_ is false, the operation has already been completed.
        // Otherwise, start the second operation.
        if (__state_.__success_)
        {
          execution::start(__op2_);
        }
      }
      _CCCL_CATCH_ALL
      {
        execution::set_error(static_cast<_Rcvr&&>(__state_.__rcvr_env_.__base()), execution::current_exception());
      }
    }

    _CCCL_DEVICE_API void __device_start() noexcept
    {
      _CCCL_ASSERT(false, "starting a sequence operation on device is not yet supported.");
      ::cuda::std::terminate();

      execution::start(__op1_);
      execution::start(__op2_);
    }

    [[nodiscard]]
    _CCCL_API static constexpr auto __mk_state(_CvSndr1& __child1, _Rcvr& __rcvr) noexcept
    {
      auto __sch = get_completion_scheduler<set_value_t>(get_env(__child1), get_env(__rcvr));
      return __state_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __env2_t{static_cast<stream_scheduler&&>(__sch)}};
    }

    __state_t<_Rcvr> __state_;
    // FUTURE: these two opstates could be stored in a variant to save space.
    connect_result_t<_CvSndr1, __rcvr1_t<_Rcvr>> __op1_;
    connect_result_t<_CvSndr2, __rcvr2_t<_Rcvr>> __op2_;
  };
};

template <class _Sndr1, class _Sndr2>
struct sequence_t::__sndr_t
{
  using sender_concept = sender_t;

  _CCCL_API constexpr explicit __sndr_t(stream_scheduler __sch, _Sndr1 __child1, _Sndr2 __child2) noexcept
      : __sch_(static_cast<stream_scheduler&&>(__sch))
      , __child1_{static_cast<_Sndr1&&>(__child1)}
      , __child2_{static_cast<_Sndr2&&>(__child2)}
  {}

  template <class _Rcvr>
  [[nodiscard]]
  _CCCL_API auto connect(_Rcvr __rcvr) && -> sequence_t::__opstate_t<_Rcvr, _Sndr1, _Sndr2>
  {
    return {static_cast<_Rcvr&&>(__rcvr), static_cast<_Sndr1&&>(__child1_), static_cast<_Sndr2&&>(__child2_)};
  }

  template <class _Rcvr>
  [[nodiscard]]
  _CCCL_API auto connect(_Rcvr __rcvr) const& -> sequence_t::__opstate_t<_Rcvr, const _Sndr1&, const _Sndr2&>
  {
    return {static_cast<_Rcvr&&>(__rcvr), __child1_, __child2_};
  }

  template <class _Self, class... _Env>
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(
      auto(__completions1) = get_child_completion_signatures<_Self, _Sndr1, __fwd_env_t<_Env>...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__completions2) = get_child_completion_signatures<_Self, _Sndr2, env<__env2_t, _Env>...>())
      {
        // __swallow_transform to ignore the first sender's value completions
        return transform_completion_signatures(__completions1, __swallow_transform{}) + __completions2
             + __eptr_completion();
      }
    }
    _CCCL_UNREACHABLE();
  }

  struct __attrs_t
  {
    [[nodiscard]]
    _CCCL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
    {
      return get_stream(__self_->__sch_);
    }

    [[nodiscard]]
    _CCCL_API constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept
    {
      return stream_domain{};
    }

    _CCCL_API constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::asynchronous;
    }

    // template <class... _Env>
    // [[nodiscard]]
    // _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>, _Env&&... __env) const noexcept
    // {
    //   return get_completion_scheduler<set_value_t>(execution::get_env(__self_->__child2_), env{__env2_t{}};
    // }

    const __sndr_t* __self_;
  };

  // TODO(ericniebler): fix the sender attributes here.
  [[nodiscard]]
  _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return __attrs_t{this};
  }

private:
  stream_scheduler __sch_;
  _Sndr1 __child1_;
  _Sndr2 __child2_;
};
} // namespace __stream

/////////////////////////////////////////////////////////////////////////////////
// sequence: customization for the stream scheduler
template <>
struct stream_domain::__apply_t<sequence_t> : __stream::sequence_t
{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_SEQUENCE
