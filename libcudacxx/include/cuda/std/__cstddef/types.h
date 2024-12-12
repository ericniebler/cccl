// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDDEF_TYPES_H
#define _LIBCUDACXX___CSTDDEF_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <cstddef>
#else
#  if !defined(offsetof)
#    define offsetof(type, member) (::size_t)((char*) &(((type*) 0)->member) - (char*) 0)
#  endif // !offsetof
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(__CLANG_MAX_ALIGN_T_DEFINED) || defined(_GCC_MAX_ALIGN_T) || defined(__DEFINED_max_align_t) \
  || defined(__NetBS)
// Re-use the compiler's <stddef.h> max_align_t where possible.
using ::max_align_t;
#else
using max_align_t = long double;
#endif

using nullptr_t = decltype(nullptr);
using ::ptrdiff_t;
using ::size_t;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDDEF_TYPES_H