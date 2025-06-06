//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// constexpr bool operator<(monostate, monostate) noexcept { return false; }
// constexpr bool operator>(monostate, monostate) noexcept { return false; }
// constexpr bool operator<=(monostate, monostate) noexcept { return true; }
// constexpr bool operator>=(monostate, monostate) noexcept { return true; }
// constexpr bool operator==(monostate, monostate) noexcept { return true; }
// constexpr bool operator!=(monostate, monostate) noexcept { return false; }
// constexpr strong_ordering operator<=>(monostate, monostate) noexcept { return strong_ordering::equal; } // since
// C++20

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using M = cuda::std::monostate;
  constexpr M m1{};
  constexpr M m2{};
  assert(testComparisons(m1, m2, /*isEqual*/ true, /*isLess*/ false));
  AssertComparisonsAreNoexcept<M>();

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  assert(testOrder(m1, m2, cuda::std::strong_ordering::equal));
  AssertOrderAreNoexcept<M>();
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
