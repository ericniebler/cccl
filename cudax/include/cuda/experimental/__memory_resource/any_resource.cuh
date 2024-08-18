//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX_MEMORY_RESOURCE_ANY_RESOURCE_H
#define _CUDAX_MEMORY_RESOURCE_ANY_RESOURCE_H

#include <cuda/__memory_resource/properties.h>

#include <cuda/experimental/__detail/basic_any.cuh>

namespace cuda::experimental
{
template <class _Self, class _Property>
struct __any_property : __interface<__any_property<_Self, _Property>>
{
  friend constexpr void get_property(_Self const& __self, _Property __prop) noexcept {}

  template <class _Ty>
  using members = __members<_Ty>;
};

template <class _Self, class... _Properties>
struct __any_resource
    : __interface<__any_resource<_Self, _Properties...>,
                  0,
                  detail::__any_equality_comparable<>,
                  __any_property<_, _Properties>...>
{};

template <class... _Properties>
struct any_resource : __basic_any<__any_resource<any_resource<_Properties...>, _Properties...>>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(any_resource);

  _CCCL_NODISCARD void* allocate(const size_t __bytes,
                                 const size_t __alignment = _CUDA_VMR::default_cuda_malloc_alignment)
  {
    constexpr auto __allocate = &any_resource::allocate;
    return __vcall<__allocate>(this, __bytes, __alignment);
  }

  void deallocate(void* __ptr, const size_t __bytes, const size_t __alignment = _CUDA_VMR::default_cuda_malloc_alignment)
  {
    constexpr auto __deallocate = &any_resource::deallocate;
    return __vcall<__deallocate>(this, __ptr, __bytes, __alignment);
  }

  template <class _Ty>
  using members = __members<_Ty, &_Ty::allocate, &_Ty::deallocate>;
};
} // namespace cuda::experimental

#endif // _CUDAX_MEMORY_RESOURCE_ANY_RESOURCE_H
