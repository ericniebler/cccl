//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_BULK
#define __CUDAX_EXECUTION_BULK

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__launch/configuration.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Pol, class _Shape, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_state_t
{
  _CCCL_NO_UNIQUE_ADDRESS _Pol __pol_;
  _Shape __shape_;
  _Fn __fn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// attributes for bulk senders
template <class _Sndr, class _Shape>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_attrs_t
{
  _CCCL_API constexpr auto query(get_launch_config_t) const noexcept
  {
    constexpr int __block_threads = 256;
    const int __grid_blocks       = (static_cast<int>(__shape_) + __block_threads - 1) / __block_threads;
    return experimental::make_config(block_dims<__block_threads>, grid_dims(__grid_blocks));
  }

  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query>)
  _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>)
    -> decltype(auto)
  {
    return execution::get_env(__sndr_).query(_Query{});
  }

  _Shape __shape_;
  const _Sndr& __sndr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// generic bulk utilities
template <class _BulkTag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_t
{
  template <class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __transform_value_completion_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      if constexpr (_CUDA_VSTD::is_same_v<_BulkTag, bulk_chunked_t>)
      {
        if constexpr (_CUDA_VSTD::__is_callable_v<_Fn&, _Shape, _Shape, _Ts&...>)
        {
          return completion_signatures<set_value_t(_Ts...)>{}
               + __eptr_completion_if<!__nothrow_callable<_Fn&, _Shape, _Shape, _Ts&...>>();
        }
        else
        {
          return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _BulkTag),
                                              _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                              _WITH_FUNCTION(_Fn),
                                              _WITH_ARGUMENTS(_Shape, _Shape, _Ts & ...)>();
        }
      }
      else if constexpr (_CUDA_VSTD::__is_callable_v<_Fn&, _Shape, _Ts&...>)
      {
        return completion_signatures<set_value_t(_Ts...)>{}
             + __eptr_completion_if<!__nothrow_callable<_Fn&, _Shape, _Ts&...>>();
      }
      else
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _BulkTag),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Fn),
                                            _WITH_ARGUMENTS(_Shape, _Ts & ...)>();
      }
    }
  };

  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __derived_opstate_t     = typename _BulkTag::template __opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;
    static_assert(!_CUDA_VSTD::is_same_v<__derived_opstate_t, __opstate_t>,
                  "The derived operation state must not be the same as the base operation state");
    using __rcvr_t = __rcvr_ref_t<__derived_opstate_t, __fwd_env_t<env_of_t<_Rcvr>>>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __shape_{__shape}
        , __fn_{static_cast<_Fn&&>(__fn)}
        , __opstate_{
            execution::connect(static_cast<_CvSndr&&>(__sndr), __ref_rcvr(*static_cast<__derived_opstate_t*>(this)))}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate_);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__rcvr_));
    }

    _Rcvr __rcvr_;
    _Shape __shape_;
    _Fn __fn_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate_;
  };

  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class _CvSndr, class _Rcvr>
    using __opstate_t = typename _BulkTag::template __opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

    template <class _Self, class... _Env>
    [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__child_completions) = execution::get_child_completion_signatures<_Self, _Sndr, _Env...>())
      {
        return transform_completion_signatures(__child_completions, __transform_value_completion_fn<_Shape, _Fn>{});
      }
    }

    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_BulkTag, bulk_t>) )
    _CCCL_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr>
    {
      return __opstate_t<_Sndr, _Rcvr>{
        static_cast<_Sndr&&>(__sndr_),
        static_cast<_Rcvr&&>(__rcvr),
        __state_.__shape_,
        static_cast<_Fn&&>(__state_.__fn_)};
    }

    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_BulkTag, bulk_t>) )
    _CCCL_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr>
    {
      return __opstate_t<const _Sndr&, _Rcvr>{__sndr_, static_cast<_Rcvr&&>(__rcvr), __state_.__shape_, __state_.__fn_};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __bulk_attrs_t<_Sndr, _Shape>
    {
      return {__state_.__shape_, __sndr_};
    }

    _CCCL_NO_UNIQUE_ADDRESS _BulkTag __tag_;
    __bulk_state_t<_Policy, _Shape, _Fn> __state_;
    _Sndr __sndr_;
  };

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t
  {
    template <class _Sndr>
    _CCCL_TRIVIAL_API friend constexpr auto operator|(_Sndr&& __sndr, __closure_t __self)
    {
      return __bulk_t{}(static_cast<_Sndr&&>(__sndr),
                        static_cast<_Policy&&>(__self.__pol_),
                        static_cast<_Shape&&>(__self.__shape_),
                        static_cast<_Fn&&>(__self.__fn_));
    }

    _CCCL_NO_UNIQUE_ADDRESS _Policy __pol_;
    _Shape __shape_;
    _Fn __fn_;
  };

  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  _CCCL_API auto operator()(_Sndr&& __sndr, _Policy __pol, _Shape __shape, _Fn __fn) const
  {
    static_assert(__is_sender<_Sndr>);
    static_assert(_CUDA_VSTD::integral<_Shape>);
    static_assert(is_execution_policy_v<_Policy>);

    using __domain_t = __early_domain_of_t<_Sndr>;
    using __sndr_t   = __bulk_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>;

    if constexpr (!dependent_sender<_Sndr>)
    {
      __assert_valid_completion_signatures(get_completion_signatures<__sndr_t>());
    }

    return transform_sender(__domain_t{},
                            __sndr_t{{}, {__pol, __shape, static_cast<_Fn&&>(__fn)}, static_cast<_Sndr&&>(__sndr)});
  }

  template <class _Policy, class _Shape, class _Fn>
  _CCCL_TRIVIAL_API auto operator()(_Policy __pol, _Shape __shape, _Fn __fn) const -> __closure_t<_Policy, _Shape, _Fn>
  {
    return {__pol, __shape, static_cast<_Fn&&>(__fn)};
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk_chunked
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_chunked_t : __bulk_t<bulk_chunked_t>
{
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  using __base_opstate_t = __bulk_t::__opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>
  {
    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>{
            static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
    {}

    template <class... _Values>
    _CCCL_API void set_value(_Values&&... __values) noexcept
    {
      _CUDAX_TRY( //
        ({
          this->__fn_(_Shape(0), _Shape(this->__shape_), __values...);
          execution::set_value(static_cast<_Rcvr&&>(this->__rcvr_), static_cast<_Values&&>(__values)...);
        }),
        _CUDAX_CATCH(...) //
        ({
          if constexpr (!__nothrow_callable<_Fn&, _Shape, _Shape, _Values&...>)
          {
            execution::set_error(static_cast<_Rcvr&&>(this->__rcvr_), ::std::current_exception());
          }
        }))
    }
  };
};

_CCCL_GLOBAL_CONSTANT auto bulk_chunked = bulk_chunked_t{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_unchunked_t : __bulk_t<bulk_unchunked_t>
{
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  using __base_opstate_t = __bulk_t::__opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>
  {
    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>{
            static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
    {}

    _CCCL_API void __check_forward_progress() const
    {
      if constexpr (_CUDA_VSTD::__is_callable_v<get_completion_scheduler_t<set_value_t>, env_of_t<_Rcvr>>)
      {
        // If the scheduler is queryable, we can check the forward progress guarantee.
        using __sched_t = _CUDA_VSTD::__call_result_t<get_completion_scheduler_t<set_value_t>, env_of_t<_Rcvr>>;
        if constexpr (__statically_queryable_with<__sched_t, get_forward_progress_guarantee_t>)
        {
          constexpr auto __guarantee = __sched_t::query(get_forward_progress_guarantee);
          static_assert(__guarantee != forward_progress_guarantee::concurrent,
                        "The default implementation does not run a scheduler with concurrent progress guarantees");
        }
        else
        {
          const auto __guarantee =
            get_forward_progress_guarantee(get_completion_scheduler<set_value_t>(execution::get_env(this->__rcvr_)));
          _CCCL_ASSERT(__guarantee != forward_progress_guarantee::concurrent,
                       "The default implementation does not run a scheduler with concurrent progress guarantees");
        }
      }
    }

    template <class... _Values>
    _CCCL_API void set_value(_Values&&... __values) noexcept
    {
      __check_forward_progress();
      _CUDAX_TRY( //
        ({
          for (_Shape __index{}; __index != this->__shape_; ++__index)
          {
            this->__fn_(_Shape(__index), __values...);
          }
          execution::set_value(static_cast<_Rcvr&&>(this->__rcvr_), static_cast<_Values&&>(__values)...);
        }),
        _CUDAX_CATCH(...) //
        ({
          if constexpr (!__nothrow_callable<_Fn&, _Shape, _Values&...>)
          {
            execution::set_error(static_cast<_Rcvr&&>(this->__rcvr_), ::std::current_exception());
          }
        }))
    }
  };

  template <class _Sndr, class _Shape, class _Fn>
  _CCCL_API auto operator()(_Sndr&& __sndr, _Shape __shape, _Fn __fn) const
  {
    const __bulk_t<bulk_unchunked_t>& __self = *this;
    return __self(static_cast<_Sndr&&>(__sndr), par, __shape, static_cast<_Fn&&>(__fn));
  }

  template <class _Shape, class _Fn>
  _CCCL_TRIVIAL_API auto operator()(_Shape __shape, _Fn __fn) const
  {
    const __bulk_t<bulk_unchunked_t>& __self = *this;
    return __self(par, __shape, static_cast<_Fn&&>(__fn));
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk_unchunked = bulk_unchunked_t{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_t : __bulk_t<bulk_t>
{
  template <class _Shape, class _Fn>
  struct __bulk_chunked_fn
  {
    template <class... _Ts>
    _CCCL_TRIVIAL_API auto operator()(_Shape __begin, _Shape __end, _Ts&&... __values) noexcept(
      __nothrow_callable<_Fn&, _Shape, decltype(__values)&...>)
    {
      for (; __begin != __end; ++__begin)
      {
        __fn_(_Shape(__begin), __values...);
      }
    }

    _Fn __fn_;
  };

  template <class _Sndr, class _Env>
  _CCCL_API static auto transform_sender(_Sndr&& __sndr, const _Env&)
  {
    auto& [__tag, __data, __child] = __sndr;
    auto& [__pol, __shape, __fn]   = __data;

    using __chunked_fn_t = __bulk_chunked_fn<decltype(__shape), decltype(__fn)>;

    // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` is customized, we will see the customization.
    return bulk_chunked(
      _CUDA_VSTD::forward_like<_Sndr>(__child), __pol, __shape, __chunked_fn_t{_CUDA_VSTD::forward_like<_Sndr>(__fn)});
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk = bulk_t{};

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__bulk_t<bulk_t>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__bulk_t<bulk_chunked_t>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__bulk_t<bulk_unchunked_t>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_BULK
