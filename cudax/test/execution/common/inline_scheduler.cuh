//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__utility/immovable.h>

#include <cuda/experimental/execution.cuh>

#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

namespace
{
template <class Domain>
struct _inline_scheduler_attrs_t
{
  template <class _Env>
  _CCCL_HOST_DEVICE static constexpr void __check_env()
  {
    if constexpr (cudax::__callable<ex::get_completion_domain_t<ex::set_value_t>, const _Env&, const _Env&>)
    {
      using actual_domain_t =
        cudax::__call_result_t<ex::get_completion_domain_t<ex::set_value_t>, const _Env&, const _Env&>;
      static_assert(cudax::__is_instantiable_with<cuda::std::common_type_t, Domain, actual_domain_t>,
                    "the specified completion domain must be convertible to the actual domain");
    }
  }

  template <class _Env>
  _CCCL_HOST_DEVICE static constexpr auto
  query(ex::get_completion_scheduler_t<ex::set_value_t>, const _Env& env) noexcept
    -> decltype(ex::get_completion_scheduler<ex::set_value_t>(env, env))
  {
    __check_env<_Env>();
    return ex::get_completion_scheduler<ex::set_value_t>(env, env);
  }

  template <class... _Env>
  _CCCL_HOST_DEVICE static constexpr auto
  query(ex::get_completion_domain_t<ex::set_value_t>, const _Env&... env) noexcept -> Domain
  {
    (__check_env<_Env>(), ...);
    return {};
  }

  _CCCL_HOST_DEVICE static constexpr auto query(ex::get_completion_behavior_t) noexcept
  {
    return ex::completion_behavior::inline_completion;
  }
};

//! Scheduler that returns a sender that always completes inline
//! (successfully).
template <class Domain = ex::default_domain>
struct inline_scheduler : _inline_scheduler_attrs_t<Domain>
{
private:
  template <class Rcvr>
  struct _opstate_t : cuda::__immovable
  {
    using operation_state_concept = ex::operation_state_t;

    _CCCL_HOST_DEVICE constexpr void start() noexcept
    {
      ex::set_value(static_cast<Rcvr&&>(_rcvr));
    }

    Rcvr _rcvr;
  };

public:
  using scheduler_concept = ex::scheduler_t;

  struct _sndr_t
  {
    using sender_concept = ex::sender_t;

    template <class Self, class... Env>
    _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
    {
      return ex::completion_signatures<ex::set_value_t()>();
    }

    template <class Rcvr>
    _CCCL_HOST_DEVICE constexpr auto connect(Rcvr rcvr) const noexcept -> _opstate_t<Rcvr>
    {
      return {{}, static_cast<Rcvr&&>(rcvr)};
    }

    _CCCL_HOST_DEVICE static constexpr auto get_env() noexcept -> _inline_scheduler_attrs_t<Domain>
    {
      return {};
    }
  };

  inline_scheduler() = default;

  _CCCL_HOST_DEVICE static constexpr _sndr_t schedule() noexcept
  {
    return {};
  }

  _CCCL_HOST_DEVICE friend bool operator==(inline_scheduler, inline_scheduler) noexcept
  {
    return true;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(inline_scheduler, inline_scheduler) noexcept
  {
    return false;
  }
};
} // namespace
