//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc> tuple(allocator_arg_t, Alloc const&)

// Libc++ has to deduce the 'allocator_arg_t' parameter for this constructor
// as 'AllocArgT'. Previously libc++ has tried to support tags derived from
// 'allocator_arg_t' by using 'is_base_of<AllocArgT, allocator_arg_t>'.
// However this breaks whenever a 2-tuple contains a reference to an incomplete
// type as its first parameter. See PR27684.

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct IncompleteType;
_CCCL_GLOBAL_VARIABLE extern IncompleteType inc1;
_CCCL_GLOBAL_VARIABLE extern IncompleteType inc2;
_CCCL_GLOBAL_VARIABLE IncompleteType const& cinc1 = inc1;
_CCCL_GLOBAL_VARIABLE IncompleteType const& cinc2 = inc2;

int main(int, char**)
{
  using IT = IncompleteType;
  { // try calling tuple(Tp const&...)
    using Tup = cuda::std::tuple<const IT&, const IT&>;
    Tup t(cinc1, cinc2);
    assert(&cuda::std::get<0>(t) == &inc1);
    assert(&cuda::std::get<1>(t) == &inc2);
  }
  { // try calling tuple(Up&&...)
    using Tup = cuda::std::tuple<const IT&, const IT&>;
    Tup t(inc1, inc2);
    assert(&cuda::std::get<0>(t) == &inc1);
    assert(&cuda::std::get<1>(t) == &inc2);
  }

  return 0;
}

struct IncompleteType
{};
_CCCL_GLOBAL_VARIABLE IncompleteType inc1;
_CCCL_GLOBAL_VARIABLE IncompleteType inc2;
