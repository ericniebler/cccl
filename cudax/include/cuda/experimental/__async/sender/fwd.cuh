//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FWD
#define __CUDAX_ASYNC_DETAIL_FWD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__async/sender/meta.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT receiver_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT operation_state_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT sender_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT scheduler_t
{};

template <class _Ty>
using __sender_concept_t = typename _CUDA_VSTD::remove_reference_t<_Ty>::sender_concept;

template <class _Ty>
using __receiver_concept_t = typename _CUDA_VSTD::remove_reference_t<_Ty>::receiver_concept;

template <class _Ty>
using __scheduler_concept_t = typename _CUDA_VSTD::remove_reference_t<_Ty>::scheduler_concept;

template <class _Ty>
inline constexpr bool __is_sender = __type_valid_v<__sender_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_receiver = __type_valid_v<__receiver_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_scheduler = __type_valid_v<__scheduler_concept_t, _Ty>;

struct _CCCL_TYPE_VISIBILITY_DEFAULT default_domain;
struct _CCCL_TYPE_VISIBILITY_DEFAULT stream_domain;
struct _CCCL_TYPE_VISIBILITY_DEFAULT dependent_sender_error;

template <class... _Sigs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT completion_signatures;

template <class _Sndr, class... _Env>
_CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto get_completion_signatures();

// handy enumerations for keeping type names readable
enum __disposition_t
{
  __value,
  __error,
  __stopped
};

struct set_value_t;
struct set_error_t;
struct set_stopped_t;
struct start_t;
struct connect_t;
struct schedule_t;
struct get_env_t;
struct sync_wait_t;

_CUDAX_GLOBAL_CONSTANT_FWD_DECL set_value_t set_value;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL set_error_t set_error;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL set_stopped_t set_stopped;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL start_t start;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL connect_t connect;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL schedule_t schedule;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL get_env_t get_env;
_CUDAX_GLOBAL_CONSTANT_FWD_DECL sync_wait_t sync_wait;

template <class _Sndr, class _Rcvr>
using connect_result_t = decltype(connect(declval<_Sndr>(), declval<_Rcvr>()));

template <class _Sndr, class... _Env>
using completion_signatures_of_t = decltype(get_completion_signatures<_Sndr, _Env...>());

template <class _Sch>
using schedule_result_t = decltype(schedule(declval<_Sch>()));

template <class _Ty>
using env_of_t = decltype(get_env(declval<_Ty>()));

template <class _Sndr, class _Rcvr>
inline constexpr bool __nothrow_connectable = noexcept(connect(declval<_Sndr>(), declval<_Rcvr>()));

namespace __detail
{
template <__disposition_t, class _Void = void>
extern _CUDA_VSTD::__undefined<_Void> __set_tag;
template <class _Void>
extern __fn_t<set_value_t>* __set_tag<__value, _Void>;
template <class _Void>
extern __fn_t<set_error_t>* __set_tag<__error, _Void>;
template <class _Void>
extern __fn_t<set_stopped_t>* __set_tag<__stopped, _Void>;
} // namespace __detail
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
