//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day;

// constexpr year_month_day
//   operator/(const year_month& ym, const day& d) noexcept;
// Returns: {ym.year(), ym.month(), d}.
//
// constexpr year_month_day
//   operator/(const year_month& ym, int d) noexcept;
// Returns: ym / day(d).
//
// constexpr year_month_day
//   operator/(const year& y, const month_day& md) noexcept;
// Returns: y / md.month() / md.day().
//
// constexpr year_month_day
//   operator/(int y, const month_day& md) noexcept;
// Returns: year(y) / md.
//
// constexpr year_month_day
//   operator/(const month_day& md, const year& y) noexcept;
// Returns: y / md.
//
// constexpr year_month_day
//   operator/(const month_day& md, int y) noexcept;
// Returns: year(y) / md.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year           = cuda::std::chrono::year;
  using month          = cuda::std::chrono::month;
  using day            = cuda::std::chrono::day;
  using year_month     = cuda::std::chrono::year_month;
  using month_day      = cuda::std::chrono::month_day;
  using year_month_day = cuda::std::chrono::year_month_day;

  constexpr month February = cuda::std::chrono::February;
  constexpr year_month Feb2018{year{2018}, February};

  { // operator/(const year_month& ym, const day& d)
    static_assert(noexcept(Feb2018 / day{2}));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(Feb2018 / day{2})>);

    static_assert((Feb2018 / day{2}).month() == February, "");
    static_assert((Feb2018 / day{2}).day() == day{2}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (int j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 28; ++k)
        {
          year y(i);
          month m(j);
          day d(k);
          year_month ym(y, m);
          year_month_day ymd = ym / d;
          assert(ymd.year() == y);
          assert(ymd.month() == m);
          assert(ymd.day() == d);
        }
      }
    }
  }

  { // operator/(const year_month& ym, int d)
    static_assert(noexcept(Feb2018 / 2));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(Feb2018 / 2)>);

    static_assert((Feb2018 / 2).month() == February, "");
    static_assert((Feb2018 / 2).day() == day{2}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (int j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 28; ++k)
        {
          year y(i);
          month m(j);
          day d(k);
          year_month ym(y, m);
          year_month_day ymd = ym / k;
          assert(ymd.year() == y);
          assert(ymd.month() == m);
          assert(ymd.day() == d);
        }
      }
    }
  }

  { // operator/(const year_month& ym, int d)
    static_assert(noexcept(Feb2018 / 2));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(Feb2018 / 2)>);

    static_assert((Feb2018 / 2).month() == February, "");
    static_assert((Feb2018 / 2).day() == day{2}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (int j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 28; ++k)
        {
          year y(i);
          month m(j);
          day d(k);
          year_month ym(y, m);
          year_month_day ymd = ym / k;
          assert(ymd.year() == y);
          assert(ymd.month() == m);
          assert(ymd.day() == d);
        }
      }
    }
  }

  { // operator/(const year& y, const month_day& md) (and switched)
    static_assert(noexcept(year{2018} / month_day{February, day{2}}));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(year{2018} / month_day{February, day{2}})>);
    static_assert(noexcept(month_day{February, day{2}} / year{2018}));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(month_day{February, day{2}} / year{2018})>);

    static_assert((year{2018} / month_day{February, day{2}}).month() == February, "");
    static_assert((year{2018} / month_day{February, day{2}}).day() == day{2}, "");
    static_assert((month_day{February, day{2}} / year{2018}).month() == February, "");
    static_assert((month_day{February, day{2}} / year{2018}).day() == day{2}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (int j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 28; ++k)
        {
          year y(i);
          month m(j);
          day d(k);
          month_day md(m, d);
          year_month_day ymd1 = y / md;
          year_month_day ymd2 = md / y;
          assert(ymd1.year() == y);
          assert(ymd2.year() == y);
          assert(ymd1.month() == m);
          assert(ymd2.month() == m);
          assert(ymd1.day() == d);
          assert(ymd2.day() == d);
          assert(ymd1 == ymd2);
        }
      }
    }
  }

  { // operator/(const month_day& md, int y) (and switched)
    static_assert(noexcept(2018 / month_day{February, day{2}}));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(2018 / month_day{February, day{2}})>);
    static_assert(noexcept(month_day{February, day{2}} / 2018));
    static_assert(cuda::std::is_same_v<year_month_day, decltype(month_day{February, day{2}} / 2018)>);

    static_assert((2018 / month_day{February, day{2}}).month() == February, "");
    static_assert((2018 / month_day{February, day{2}}).day() == day{2}, "");
    static_assert((month_day{February, day{2}} / 2018).month() == February, "");
    static_assert((month_day{February, day{2}} / 2018).day() == day{2}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (int j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 28; ++k)
        {
          year y(i);
          month m(j);
          day d(k);
          month_day md(m, d);
          year_month_day ymd1 = i / md;
          year_month_day ymd2 = md / i;
          assert(ymd1.year() == y);
          assert(ymd2.year() == y);
          assert(ymd1.month() == m);
          assert(ymd2.month() == m);
          assert(ymd1.day() == d);
          assert(ymd2.day() == d);
          assert(ymd1 == ymd2);
        }
      }
    }
  }

  return 0;
}
