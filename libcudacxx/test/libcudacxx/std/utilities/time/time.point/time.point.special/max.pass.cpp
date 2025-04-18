//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// static constexpr time_point max(); // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration;
  typedef cuda::std::chrono::time_point<Clock, Duration> TP;
  static_assert(noexcept(TP::max()));
#if TEST_STD_VER > 2017
  static_assert(noexcept(TP::max()));
#endif
  assert(TP::max() == TP(Duration::max()));

  return 0;
}
