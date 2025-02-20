//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// class month_day_last;

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_day_last = cuda::std::chrono::month_day_last;

  static_assert(cuda::std::is_trivially_copyable_v<month_day_last>, "");
  static_assert(cuda::std::is_standard_layout_v<month_day_last>, "");

  return 0;
}
