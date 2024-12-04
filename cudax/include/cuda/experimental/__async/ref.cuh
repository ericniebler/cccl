//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_REF
#define __CUDAX_ASYNC_DETAIL_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_const.h>

#include <cuda/experimental/__async/stream_ref.cuh>

namespace cuda::experimental
{
// clang-format off
template <class _Ty>
_CCCL_CONCEPT __has_operator_arrow =
  _CCCL_REQUIRES_EXPR((_Ty), _Ty&&(*__ty)())
  (
    __ty().operator->()
  );
// clang-format on

// A non-owning reference to either a future or a raw value. Useful as the
// argument of APIs to permit the user to pass either a future or a value.
template <class _CvValue>
struct ref : __basic_future<ref<_CvValue>>
{
  using value_type _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_CvValue>;

  using __cv_future_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::__maybe_const<_CUDA_VSTD::is_const_v<_CvValue>, future<value_type>>;

  _CUDAX_HOST_API ref(__cv_future_t& __fut) noexcept
      : __basic_future<ref>(__fut.get_stream())
      , __value_ptr_(_CUDA_VSTD::addressof(__fut.get_unsynchronized()))
  {}

  _CUDAX_HOST_API ref(_CvValue& __val) noexcept
      : __basic_future<ref>(detail::__invalid_stream)
      , __value_ptr_(_CUDA_VSTD::addressof(__val))
  {}

  _CUDAX_HOST_API auto get_unsynchronized() const noexcept -> _CvValue&
  {
    return *__value_ptr_;
  }

  _CUDAX_HOST_API auto operator->() const noexcept -> _CvValue const&
  {
    return *__value_ptr_;
  }

  _CCCL_TEMPLATE(class _CvRefValue = _CvValue const&)
  _CCCL_REQUIRES(__has_operator_arrow<_CvRefValue>)
  _CUDAX_HOST_API decltype(auto) get() const noexcept
  {
    return *const_cast<_CvValue const&>(*__value_ptr_).operator->();
  }

private:
  _CvValue* __value_ptr_;
};

template <class _Value>
ref(_Value&) -> ref<_Value>;

template <class _Value>
ref(_Value const&) -> ref<_Value const>;

template <class _Value>
ref(future<_Value>&) -> ref<_Value>;

template <class _Value>
ref(const future<_Value>&) -> ref<_Value const>;

#if _CCCL_STD_VER >= 2020

template <class _Value>
using cref = ref<_Value const>;

#else // _CCCL_STD_VER >= 2020

template <class _Value>
struct cref : ref<_Value const>
{
  using ref<_Value const>::ref;
  cref(_Value const&&) = delete;
  cref(ref<_Value> other) noexcept
      : ref<_Value const>(other)
  {}
};

template <class _Value>
cref(_Value const&) -> cref<_Value>;

template <class _Value>
cref(const future<_Value>&) -> cref<_Value>;

#endif // _CCCL_STD_VER >= 2020

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_DETAIL_REF
