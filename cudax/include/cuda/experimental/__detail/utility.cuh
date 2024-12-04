//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_UTILITY_H
#define __CUDAX_DETAIL_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/preprocessor.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/experimental/__detail/config.cuh>

namespace cuda::experimental
{
// Classes can inherit from this type to become immovable.
struct __immovable
{
  __immovable()                         = default;
  __immovable(__immovable&&)            = delete;
  __immovable& operator=(__immovable&&) = delete;
};

template <class... _Types>
struct _CCCL_DECLSPEC_EMPTY_BASES __inherit : _Types...
{};

template <class _Type, template <class...> class _Template>
inline constexpr bool __is_specialization_of = false;

template <template <class...> class _Template, class... _Args>
inline constexpr bool __is_specialization_of<_Template<_Args...>, _Template> = true;

template <class _Tp>
using __identity_t _CCCL_NODEBUG_ALIAS = _Tp;

using _CUDA_VSTD::declval;

struct uninit_t
{
  explicit uninit_t() = default;
};

_CCCL_GLOBAL_CONSTANT uninit_t uninit{};

template <template <class...> class _Cy, class... _Args>
_CCCL_CONCEPT __is_instantiable_with = _CUDA_VSTD::_IsValidExpansion<_Cy, _Args...>::value;

template <class _From, class _To>
_CCCL_CONCEPT __decays_to = _CUDA_VSTD::same_as<_CUDA_VSTD::decay_t<_From>, _To>;

template <class _From, class _To>
_CCCL_CONCEPT __decays_to_derived_from = _CUDA_VSTD::derived_from<_CUDA_VSTD::decay_t<_From>, _To>;

template <class... _Types>
_CCCL_CONCEPT __decay_copyable = //
  (_CUDA_VSTD::constructible_from<_CUDA_VSTD::decay_t<_Types>, _Types> && ...);

template <class _Type>
_CCCL_CONCEPT __movable_value = //
  __decay_copyable<_Type> && _CUDA_VSTD::move_constructible<_CUDA_VSTD::decay_t<_Type>>;

struct __decay_xvalue_fn
{
  template <class _Arg>
  _CUDAX_TRIVIAL_API _Arg operator()(_Arg&& __arg) const noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Arg>)
  {
    return static_cast<_Arg&&>(__arg);
  }
};

_CCCL_GLOBAL_CONSTANT __decay_xvalue_fn __decay_xvalue{};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_UTILITY_H
