//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FUTURE_FWD
#define __CUDAX_ASYNC_DETAIL_FUTURE_FWD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/tuple>
#include <cuda/stream_ref>

#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>

namespace cuda::experimental
{
struct stream_ref;

struct __future_base;

template <class _Future>
struct __basic_future;

template <class _Value>
struct future;

template <class _Value>
struct shared_future;

template <class _CvValue>
struct ref;

template <class _Type>
inline constexpr bool __cudax_ref_impl = false;

template <class _CvValue>
inline constexpr bool __cudax_ref_impl<ref<_CvValue>> = true;

#if _CCCL_STD_VER >= 2020

template <class _Value>
using cref = ref<_Value const>;

#else

template <class _Value>
struct cref;

template <class _Value>
inline constexpr bool __cudax_ref_impl<cref<_Value>> = true;

#endif

template <class _Type>
inline constexpr bool __cudax_ref = __cudax_ref_impl<_CUDA_VSTD::decay_t<_Type>>;

struct __maybe_enqueue_fn;
struct __relocatable_value_fn;

namespace __get_unsync_ns
{
struct __fn;
} // namespace __get_unsync_ns

namespace __async_transform_ns
{
struct __fn;
} // namespace __async_transform_ns

inline namespace __cpo
{
extern const __get_unsync_ns::__fn get_unsynchronized;
extern const __async_transform_ns::__fn async_transform;
extern const __maybe_enqueue_fn __maybe_enqueue;
extern const __relocatable_value_fn relocatable_value;
} // namespace __cpo

template <class _Value>
using __maybe_enqueue_result_t = decltype(__decay_xvalue(__maybe_enqueue(declval<_Value>(), declval<stream_ref>())));

template <class _Task>
using __task_value_of = decltype(__decay_xvalue(declval<_Task>().enqueue(declval<stream_ref>())));

template <class _Future>
using __wait_result_t = decltype(__decay_xvalue(get_unsynchronized(declval<_Future>())));

template <typename _Arg>
using __async_transform_result_t =
  decltype(__decay_xvalue(async_transform(declval<::cuda::stream_ref>(), declval<_Arg>())));

template <typename _Arg>
using __relocatable_value_result_t = decltype(__decay_xvalue(relocatable_value(declval<_Arg>())));

template <typename _Arg>
using relocatable_value_of_t = _CUDA_VSTD::decay_t<__relocatable_value_result_t<__async_transform_result_t<_Arg>>>;

struct __future_access
{
  static _CUDAX_TRIVIAL_HOST_API stream_ref __exchange_stream(const __future_base& __fut, stream_ref __stream);

  template <class _Task>
  static _CUDAX_TRIVIAL_HOST_API auto __make_future_from_task(_Task __task, stream_ref __stream);

  template <class _Value>
  static _CUDAX_TRIVIAL_HOST_API auto __make_future_from_value(_Value __value, stream_ref __stream);

  template <class _Value>
  static _CUDAX_TRIVIAL_HOST_API auto __make_shared_future(future<_Value>&& __fut) -> shared_future<_Value>;
};

// clang-format off
template <class _Type>
_CCCL_CONCEPT Value = true;

template <class _Type>
_CCCL_CONCEPT Future =
  __movable_value<_Type> &&
  _CCCL_REQUIRES_EXPR((_Type), _Type&&(*__fut)())
  (
    typename(typename _CUDA_VSTD::remove_reference_t<_Type>::value_type),
    _Same_as(__cudax::stream_ref) __fut().get_stream(),
    _Same_as(void) __fut().wait(),
    _Same_as(bool) __fut().valid()
  );

template <class _Type>
_CCCL_CONCEPT Task =
  __movable_value<_Type> &&
  _CCCL_REQUIRES_EXPR((_Type), _CUDA_VSTD::decay_t<_Type>&& __value, stream_ref const& __stream)
  (
    (_CUDA_VSTD::move(__value).enqueue(__stream))
  );
// clang-format on

template <class _Type>
_CUDAX_HOST_API decltype(auto)
__async_result_of(_Type && (*__value)() = nullptr, const stream_ref& (*__stream)() = nullptr)
{
  if constexpr (Future<_Type>)
  {
    return get_unsynchronized(__value());
  }
  else if constexpr (Task<_Type>)
  {
    if constexpr (Future<__task_value_of<_Type>>)
    {
      return get_unsynchronized(__value().enqueue(__stream()));
    }
    else
    {
      return __value().enqueue(__stream());
    }
  }
  else
  {
    return __value();
  }
}

// For a future<T>, returns T. For an task, returns the result of enqueueing
// the task. For all other types, returns the type itself.
template <class _Tp>
using async_result_of_t = decltype(__decay_xvalue(__cudax::__async_result_of<_Tp>()));

extern const struct __wait_t wait;
extern const struct __wait_all_t wait_all;
extern const struct __fuse_t fuse;
extern const struct __sequence_t sequence;

namespace __async
{
template <class _Is, class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(__tupl<_Is, _Values...>&& __tupl);
template <class _Is, class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(__tupl<_Is, _Values...>& __tupl);
template <class _Is, class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(const __tupl<_Is, _Values...>& __tupl);
} // namespace __async

} // namespace cuda::experimental

_LIBCUDACXX_BEGIN_NAMESPACE_STD
template <class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(tuple<_Values...>&& __tupl);
template <class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(tuple<_Values...>& __tupl);
template <class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(const tuple<_Values...>& __tupl);
_LIBCUDACXX_END_NAMESPACE_STD

#endif // __CUDAX_ASYNC_DETAIL_FUTURE_FWD
