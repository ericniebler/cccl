//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// default ctor

#include <cuda/std/bitset>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero
template <cuda::std::size_t N>
__host__ __device__ constexpr void test_default_ctor()
{
  {
    constexpr cuda::std::bitset<N> v1;
    assert(v1.size() == N);
    for (cuda::std::size_t i = 0; i < v1.size(); ++i)
    {
      {
        assert(v1[i] == false);
      }
    }
  }
#if TEST_STD_VER >= 11
  {
    constexpr cuda::std::bitset<N> v1;
    static_assert(v1.size() == N, "");
  }
#endif
}

__host__ __device__ constexpr bool test()
{
  test_default_ctor<0>();
  test_default_ctor<1>();
  test_default_ctor<31>();
  test_default_ctor<32>();
  test_default_ctor<33>();
  test_default_ctor<63>();
  test_default_ctor<64>();
  test_default_ctor<65>();
  test_default_ctor<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
