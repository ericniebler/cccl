//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_FORK
#define __CUDAX_EXECUTION_FORK

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/pod_tuple.h>
#include <cuda/std/atomic>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/variant.cuh>

#include <array> // IWYU pragma: keep for ::std::tuple_size

#include <cuda/experimental/__execution/prologue.cuh>

#if defined(_CCCL_BUILTIN_BIT_CAST)
#  define _CUDAX_BIT_CAST_CONSTEXPR constexpr
#else
#  define _CUDAX_BIT_CAST_CONSTEXPR inline
#endif

namespace cuda::experimental::execution
{
struct fork_t
{
  template <class _Sndr, class _Env>
  struct __tuple_t;

  struct __waiter_t
  {
    using __notify_fn = void(__waiter_t*) noexcept;

    _CCCL_API constexpr void __notify() noexcept
    {
      __notify_(this);
    }

    __waiter_t* __next_{nullptr};
    __notify_fn* __notify_{nullptr};
  };

  _CCCL_TRIVIAL_API static _CUDAX_BIT_CAST_CONSTEXPR auto __started() noexcept -> __waiter_t*
  {
    return ::cuda::std::bit_cast<__waiter_t*>(_CUDA_VSTD::uintptr_t(alignof(__waiter_t)));
  }

  _CCCL_TRIVIAL_API static _CUDAX_BIT_CAST_CONSTEXPR auto __completed() noexcept -> __waiter_t*
  {
    return ::cuda::std::bit_cast<__waiter_t*>(_CUDA_VSTD::uintptr_t(2 * alignof(__waiter_t)));
  }

  struct __visitor
  {
    struct __materialize
    {
      template <class _Rcvr, class _Tag, class... _Args>
      _CCCL_TRIVIAL_API constexpr void operator()(_Rcvr& __rcvr, _Tag, _Args const&... __args) const noexcept
      {
        _Tag{}(static_cast<_Rcvr&&>(__rcvr), __args...);
      }

      template <class _Rcvr>
      _CCCL_TRIVIAL_API constexpr void
      operator()(_Rcvr& __rcvr, set_error_t, ::std::exception_ptr const& __eptr) const noexcept
      {
        // We promised in the completion signatures that we would send an rvalue
        // exception_ptr, so we need to send a copy to the receiver.
        execution::set_error(static_cast<_Rcvr&&>(__rcvr), ::std::exception_ptr{__eptr});
      }
    };

    template <class _Rcvr, class _Tuple>
    _CCCL_TRIVIAL_API constexpr void operator()(_Rcvr& __rcvr, _Tuple const& __tupl) const noexcept
    {
      _CUDA_VSTD::__apply(__materialize{}, __tupl, __rcvr);
    }
  };

  template <class _Env, class _Variant>
  struct __state_t
  {
    template <class _Rcvr>
    _CCCL_API constexpr void __complete(_Rcvr& __rcvr) noexcept
    {
      __results_.__visit(__visitor{}, __results_, __rcvr);
    }

    _Env __env_{};
    _Variant __results_{};
    _CUDA_VSTD::atomic<__waiter_t*> __waiters_{nullptr};
  };

  template <class _Tag, class... _Args>
  using __cref_sig_t = _Tag(_CUDA_VSTD::decay_t<_Args> const&...);

  template <class _Sndr, class _Env>
  _CCCL_API static _CCCL_CONSTEVAL auto __completions()
  {
    _CUDAX_LET_COMPLETIONS(auto(__completions) = execution::get_completion_signatures<_Sndr, _Env>())
    {
      using __completions_t = decltype(__completions);
      if constexpr (!__completions_t::__decay_copyable::value)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, fork_t),
                                            _WHAT(_PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE),
                                            _CUDA_VSTD::__type_apply_q<_WITH_COMPLETIONS, __completions_t>>();
      }
      else
      {
        using __result_t =
          typename __completions_t::template __transform<_CUDA_VSTD::__type_indirect_quote<__cref_sig_t>>;
        constexpr bool __nothrow = __completions_t::__nothrow_decay_copyable::value;
        return __result_t{} + __eptr_completion_if<!__nothrow>();
      }
    }
  }

  template <class _Sndr, class _Env>
  using __completions_t = decltype(__completions<_Sndr, _Env>());

  template <class _Sndr, class _Env, class _Rcvr>
  struct __opstate_t : __waiter_t
  {
    using operation_state_concept = operation_state_t;

    _CCCL_API constexpr explicit __opstate_t(_Rcvr&& __rcvr, __tuple_t<_Sndr, _Env>& __opstate)
        : __waiter_t{nullptr, &__opstate_t::__notify}
        , __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __opstate_(__opstate)
    {}

    _CCCL_API static void __notify(__waiter_t* __waiter) noexcept
    {
      auto* const __self = static_cast<__opstate_t*>(__waiter);
      __self->__opstate_.__state_.__complete(__self->__rcvr_);
    }

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API _CUDAX_BIT_CAST_CONSTEXPR void start() noexcept
    {
      // Start the fork operation state if it isn't already started:
      __opstate_.__start();

      // If the fork operation state is already completed, we can immediately forward the
      // result to the receiver. Otherwise, we need to add ourselves to the waiters list.
      auto* __expected = __opstate_.__state_.__waiters_.load(_CUDA_VSTD::memory_order_relaxed);
      if (__expected == __completed())
      {
        return __opstate_.__state_.__complete(__rcvr_);
      }

      // We must add ourselves to the front of the waiters list so that we can be notified
      // when the operation completes.
      __next_ = __expected;
      while (!__opstate_.__state_.__waiters_.compare_exchange_weak(__expected, this, _CUDA_VSTD::memory_order_acq_rel))
      {
        if (__expected == __completed())
        {
          // The operation completed while we were trying to add ourselves to the waiters
          // list. We can forward the results to the receiver and return.
          return __opstate_.__state_.__complete(__rcvr_);
        }
        // The head of the waiters list was changed by another thread, so we need to
        // update our next pointer and try again.
        __next_ = __expected;
      }
    }

    _Rcvr __rcvr_;
    __tuple_t<_Sndr, _Env>& __opstate_;
  };

  template <class _Sndr, class _Env>
  struct __sndr_t;

  template <class _Sndr, class _Env>
  struct __tuple_t
  {
    using operation_state_concept = operation_state_t;
    using __child_completions_t   = completion_signatures_of_t<_Sndr, _Env>;
    using __results_t =
      typename __completions_t<_Sndr, _Env>::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;

    _CCCL_API constexpr explicit __tuple_t(_Sndr&& __sndr, _Env __env)
        : __state_{static_cast<_Env&&>(__env)}
        , __opstate_(execution::connect(static_cast<_Sndr&&>(__sndr), __ref_rcvr(*this)))
    {}

    _CCCL_IMMOVABLE_OPSTATE(__tuple_t);

    _CCCL_API constexpr void __start() noexcept
    {
      // Start the child operation state exactly once:
      __waiter_t* __expected = nullptr;
      if (__state_.__waiters_.compare_exchange_strong(__expected, __started(), _CUDA_VSTD::memory_order_acq_rel))
      {
        execution::start(__opstate_);
      }
    }

    template <class _Tag, class... _Args>
    _CCCL_API _CUDAX_BIT_CAST_CONSTEXPR void __complete(_Tag, _Args&&... __args) noexcept
    {
      // Write the results into the results variant, catching any exceptions that may be thrown
      // during the copy or move of the arguments. If an exception is thrown, we set an error
      // completion signature instead.
      _CCCL_TRY
      {
        using __tuple_t = _CUDA_VSTD::__tuple<_Tag, _Args...>;
        __state_.__results_.template __emplace<__tuple_t>(_Tag{}, static_cast<_Args&&>(__args)...);
      }
      _CCCL_CATCH_ALL
      {
        if constexpr (!__child_completions_t::__nothrow_decay_copyable::value)
        {
          using __tuple_t = _CUDA_VSTD::__tuple<set_error_t, ::std::exception_ptr>;
          __state_.__results_.template __emplace<__tuple_t>(set_error_t{}, ::std::current_exception());
        }
      }

      // Notify all waiters that the results are ready.
      auto* __waiters = __state_.__waiters_.exchange(__completed(), _CUDA_VSTD::memory_order_acq_rel);
      while (__waiters != __started())
      {
        _CUDA_VSTD::exchange(__waiters, __waiters->__next_)->__notify();
      }
    }

    template <class... _Values>
    _CCCL_API constexpr void set_value(_Values&&... __values) noexcept
    {
      __complete(set_value_t{}, static_cast<_Values&&>(__values)...);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __error) noexcept
    {
      __complete(set_error_t{}, static_cast<_Error&&>(__error));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      __complete(set_stopped_t{});
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> _Env const&
    {
      return __state_.__env_;
    }

    template <size_t _Ip>
    [[nodiscard]] _CCCL_API constexpr auto get() const noexcept -> __sndr_t<_Sndr, _Env> const&
    {
      return __elem_;
    }

    __state_t<_Env, __results_t> __state_;
    __sndr_t<_Sndr, _Env> __elem_{{}, {}, *this};
    connect_result_t<_Sndr, __rcvr_ref_t<__tuple_t, _Env const&>> __opstate_;
  };

  template <class _Sndr, class _Env = env<>>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr, _Env __env = {}) const
  {
    return __tuple_t<_Sndr, _Env>{static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env)};
  }
};

template <class _Sndr, class _Env>
struct fork_t::__sndr_t : __move_only
{
  using sender_concept = sender_t;

  template <class, class... _Env2>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return __completions<_Sndr, _Env>();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const -> __opstate_t<_Sndr, _Env, _Rcvr>
  {
    return __opstate_t<_Sndr, _Env, _Rcvr>{static_cast<_Rcvr&&>(__rcvr), __opstate_};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> _Env const&
  {
    return __opstate_.__state_.__env_;
  }

  _CCCL_NO_UNIQUE_ADDRESS fork_t __tag_{};
  __tuple_t<_Sndr, _Env>& __opstate_;
};

_CCCL_GLOBAL_CONSTANT fork_t fork{};

template <class _Sndr, class _Env>
inline constexpr size_t structured_binding_size<fork_t::__tuple_t<_Sndr, _Env>> = 2;

} // namespace cuda::experimental::execution

namespace std
{
template <class _Sndr, class _Env>
struct tuple_size<::cuda::experimental::execution::fork_t::__tuple_t<_Sndr, _Env>> : integral_constant<size_t, 1>
{};

template <size_t _Ip, class _Sndr, class _Env>
struct tuple_element<_Ip, ::cuda::experimental::execution::fork_t::__tuple_t<_Sndr, _Env>>
{
  using type = ::cuda::experimental::execution::fork_t::__sndr_t<_Sndr, _Env> const&;
};
} // namespace std

#undef _CUDAX_BIT_CAST_CONSTEXPR

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_FORK
