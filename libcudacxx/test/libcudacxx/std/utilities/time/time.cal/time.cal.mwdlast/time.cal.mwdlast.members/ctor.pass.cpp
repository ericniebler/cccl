//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_weekday_last;

//  constexpr month_weekday_last(const chrono::month& m,
//                               const chrono::weekday_last& wdl) noexcept;
//
//  Effects:  Constructs an object of type month_weekday_last by
//            initializing m_ with m, and wdl_ with wdl.
//
//     constexpr chrono::month        month() const noexcept;
//     constexpr chrono::weekday_last weekday_last()  const noexcept;
//     constexpr bool                 ok()    const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month              = cuda::std::chrono::month;
  using weekday            = cuda::std::chrono::weekday;
  using weekday_last       = cuda::std::chrono::weekday_last;
  using month_weekday_last = cuda::std::chrono::month_weekday_last;

  constexpr month January   = cuda::std::chrono::January;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(noexcept(month_weekday_last{January, weekday_last{Tuesday}}));

  //  bad month
  constexpr month_weekday_last mwdl1{month{}, weekday_last{Tuesday}};
  static_assert(mwdl1.month() == month{}, "");
  static_assert(mwdl1.weekday_last() == weekday_last{Tuesday}, "");
  static_assert(!mwdl1.ok(), "");

  //  bad weekday_last
  constexpr month_weekday_last mwdl2{January, weekday_last{weekday{16}}};
  static_assert(mwdl2.month() == January, "");
  static_assert(mwdl2.weekday_last() == weekday_last{weekday{16}}, "");
  static_assert(!mwdl2.ok(), "");

  //  Good month and weekday_last
  constexpr month_weekday_last mwdl3{January, weekday_last{weekday{4}}};
  static_assert(mwdl3.month() == January, "");
  static_assert(mwdl3.weekday_last() == weekday_last{weekday{4}}, "");
  static_assert(mwdl3.ok(), "");

  return 0;
}
