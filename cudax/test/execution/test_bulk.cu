//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//                       Copyright (c) 2022 Lucian Radu Teodorescu
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__numeric/iota.h>
#include <cuda/std/array>

#include <cuda/experimental/execution.cuh>

#include "common/checked_receiver.cuh"
#include "common/error_scheduler.cuh"
#include "common/inline_scheduler.cuh"
#include "common/utility.cuh"
#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

#if !defined(__CUDA_ARCH__)
using _exception_ptr = ::std::exception_ptr;
#else
struct _exception_ptr
{};
#endif

namespace
{
template <class Shape, int N>
_CCCL_HOST_DEVICE void function(Shape i, int (*counter)[N])
{
  (*counter)[i]++;
}

template <class Shape, int N>
_CCCL_HOST_DEVICE void function_range(Shape b, Shape e, int (*counter)[N])
{
  while (b != e)
  {
    (*counter)[b++]++;
  }
}

template <class Shape>
struct function_object_t
{
  int* counter_;

  _CCCL_HOST_DEVICE void operator()(Shape i)
  {
    counter_[i]++;
  }
};

template <class Shape>
struct function_object_range_t
{
  int* counter_;

  _CCCL_HOST_DEVICE void operator()(Shape b, Shape e)
  {
    while (b != e)
    {
      counter_[b++]++;
    }
  }
};

struct ignore_lvalue_ref
{
  template <class T>
  _CCCL_HOST_DEVICE ignore_lvalue_ref(T&) noexcept
  {
    // Do nothing, just ignore the value
  }
};

C2H_TEST("bulk returns a sender", "[adaptors][bulk]")
{
  auto sndr = ex::bulk(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  (void) sndr;
}

TEST_CASE("bulk_chunked returns a sender", "[adaptors][bulk]")
{
  auto sndr = ex::bulk_chunked(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int, int) {});
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  (void) sndr;
}

TEST_CASE("bulk_unchunked returns a sender", "[adaptors][bulk]")
{
  auto sndr = ex::bulk_unchunked(ex::just(19), 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  (void) sndr;
}

TEST_CASE("bulk with environment returns a sender", "[adaptors][bulk]")
{
  auto sndr = ex::bulk(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender_in<decltype(sndr), ex::env<>>);
  (void) sndr;
}

TEST_CASE("bulk_chunked with environment returns a sender", "[adaptors][bulk]")
{
  auto sndr = ex::bulk_chunked(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int, int) {});
  STATIC_REQUIRE(ex::sender_in<decltype(sndr), ex::env<>>);
  (void) sndr;
}

TEST_CASE("bulk_unchunked with environment returns a sender", "[adaptors][bulk]")
{
  auto sndr = ex::bulk_unchunked(ex::just(19), 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender_in<decltype(sndr), ex::env<>>);
  (void) sndr;
}

TEST_CASE("bulk can be piped", "[adaptors][bulk]")
{
  auto sndr = ex::just() //
            | ex::bulk(ex::par, 42, [] _CCCL_HOST_DEVICE(int) {});
  (void) sndr;
}

TEST_CASE("bulk_chunked can be piped", "[adaptors][bulk]")
{
  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, 42, [] _CCCL_HOST_DEVICE(int, int) {});
  (void) sndr;
}

TEST_CASE("bulk_unchunked can be piped", "[adaptors][bulk]")
{
  auto sndr = ex::just() //
            | ex::bulk_unchunked(42, [] _CCCL_HOST_DEVICE(int) {});
  (void) sndr;
}

TEST_CASE("bulk keeps values_type from input sender", "[adaptors][bulk]")
{
  constexpr int n = 42;
  check_value_types<types<>>(ex::just() //
                             | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) {}));
  check_value_types<types<double>>(ex::just(4.2) //
                                   | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int, double) {}));
  check_value_types<types<double, string>>(ex::just(4.2, string{}) //
                                           | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int, double, string) {}));
}

TEST_CASE("bulk_chunked keeps values_type from input sender", "[adaptors][bulk]")
{
  constexpr int n = 42;
  check_value_types<types<>>(ex::just() //
                             | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {}));
  check_value_types<types<double>>(ex::just(4.2) //
                                   | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int, double) {}));
  check_value_types<types<double, string>>(
    ex::just(4.2, string{}) | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int, double, string) {}));
}

TEST_CASE("bulk_unchunked keeps values_type from input sender", "[adaptors][bulk]")
{
  constexpr int n = 42;
  check_value_types<types<>>(ex::just() //
                             | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) {}));
  check_value_types<types<double>>(ex::just(4.2) //
                                   | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int, double) {}));
  check_value_types<types<double, string>>(ex::just(4.2, string{}) //
                                           | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int, double, string) {}));
}

TEST_CASE("bulk keeps error_types from input sender", "[adaptors][bulk]")
{
  constexpr int n = 42;
  inline_scheduler sched1{};
  error_scheduler<_exception_ptr> sched2{};
  error_scheduler<int> sched3{43};

#if !_CCCL_COMPILER(MSVC)
  // MSVCBUG https://developercommunity.visualstudio.com/t/noexcept-expression-in-lambda-template-n/10718680
  check_error_types<>(ex::just() //
                      | ex::continues_on(sched1) //
                      | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<_exception_ptr>(
    ex::just() //
    | ex::continues_on(sched2) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(ex::just_error(n) //
                         | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
#  if !defined(__CUDA_ARCH__)
  check_error_types<::std::exception_ptr, int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) {
        throw std::logic_error{"err"};
      }));
#  else
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) {
        cuda::std::__cccl_terminate();
      }));
#  endif
#endif
}

TEST_CASE("bulk_chunked keeps error_types from input sender", "[adaptors][bulk]")
{
  constexpr int n = 42;
  inline_scheduler sched1{};
  error_scheduler<_exception_ptr> sched2{};
  error_scheduler<int> sched3{43};

  check_error_types<>(ex::just() //
                      | ex::continues_on(sched1) //
                      | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
  check_error_types<_exception_ptr>(
    ex::just() //
    | ex::continues_on(sched2) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
  check_error_types<int>(ex::just_error(n) //
                         | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
#if !defined(__CUDA_ARCH__)
  check_error_types<::std::exception_ptr, int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {
        throw std::logic_error{"err"};
      }));
#else
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {
        cuda::std::__cccl_terminate();
      }));
#endif
}

TEST_CASE("bulk_unchunked keeps error_types from input sender", "[adaptors][bulk]")
{
  constexpr int n = 42;
  inline_scheduler sched1{};
  error_scheduler<_exception_ptr> sched2{};
  error_scheduler<int> sched3{43};

  check_error_types<>(ex::just() //
                      | ex::continues_on(sched1) //
                      | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<_exception_ptr>(
    ex::just() //
    | ex::continues_on(sched2) //
    | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(ex::just_error(n) //
                         | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
#if !defined(__CUDA_ARCH__)
  check_error_types<::std::exception_ptr, int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) {
        throw std::logic_error{"err"};
      }));
#else
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) {
        cuda::std::__cccl_terminate();
      }));
#endif
}

TEST_CASE("bulk can be used with a function", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter1[n]{};
  _CUDA_VSTD::fill_n(counter1, n, 0);

  auto sndr = ex::just(&counter1) //
            | ex::bulk(ex::par, n, function<int, n>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{&counter1});
  ex::start(op);

  for (int i : counter1)
  {
    CHECK(i == 1);
  }
}

TEST_CASE("bulk_chunked can be used with a function", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter2[n]{};
  _CUDA_VSTD::fill_n(counter2, n, 0);

  auto sndr = ex::just(&counter2) //
            | ex::bulk_chunked(ex::par, n, function_range<int, n>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{&counter2});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter2[i] == 1);
  }
}

TEST_CASE("bulk_unchunked can be used with a function", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter3[n]{};
  _CUDA_VSTD::fill_n(counter3, n, 0);

  auto sndr = ex::just(&counter3) //
            | ex::bulk_unchunked(n, function<int, n>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{&counter3});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter3[i] == 1);
  }
}

TEST_CASE("bulk can be used with a function object", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter[n]{0};
  function_object_t<int> fn{counter};

  auto sndr = ex::just() //
            | ex::bulk(ex::par, n, fn);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i : counter)
  {
    CHECK(i == 1);
  }
}

TEST_CASE("bulk_chunked can be used with a function object", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter[n]{0};
  function_object_range_t<int> fn{counter};

  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, n, fn);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

TEST_CASE("bulk_unchunked can be used with a function object", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter[n]{0};
  function_object_t<int> fn{counter};

  auto sndr = ex::just() //
            | ex::bulk_unchunked(n, fn);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

#if !defined(__CUDA_ARCH__)
template <class>
struct undef;

TEST_CASE("bulk can be used with a lambda", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter[n]{0};

  auto sndr = ex::just() //
            | ex::bulk(ex::par, n, [&](int i) {
                counter[i]++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i : counter)
  {
    CHECK(i == 1);
  }
}

TEST_CASE("bulk_chunked can be used with a lambda", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter[n]{0};

  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, n, [&](int b, int e) {
                while (b < e)
                {
                  counter[b++]++;
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

TEST_CASE("bulk_unchunked can be used with a lambda", "[adaptors][bulk]")
{
  constexpr int n = 9;
  int counter[n]{0};

  auto sndr = ex::just() //
            | ex::bulk_unchunked(n, [&](int i) {
                counter[i]++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}
#endif // !defined(__CUDA_ARCH__)

TEST_CASE("bulk works with all standard execution policies", "[adaptors][bulk]")
{
  auto snd1 = ex::just() //
            | ex::bulk(ex::seq, 9, [] _CCCL_HOST_DEVICE(int) {});
  auto snd2 = ex::just() //
            | ex::bulk(ex::par, 9, [] _CCCL_HOST_DEVICE(int) {});
  auto snd3 = ex::just() //
            | ex::bulk(ex::par_unseq, 9, [] _CCCL_HOST_DEVICE(int) {});
  auto snd4 = ex::just() //
            | ex::bulk(ex::unseq, 9, [] _CCCL_HOST_DEVICE(int) {});

  STATIC_REQUIRE(ex::sender<decltype(snd1)>);
  STATIC_REQUIRE(ex::sender<decltype(snd2)>);
  STATIC_REQUIRE(ex::sender<decltype(snd3)>);
  STATIC_REQUIRE(ex::sender<decltype(snd4)>);
  (void) snd1;
  (void) snd2;
  (void) snd3;
  (void) snd4;
}

TEST_CASE("bulk_chunked works with all standard execution policies", "[adaptors][bulk]")
{
  auto snd1 = ex::just() //
            | ex::bulk_chunked(ex::seq, 9, [] _CCCL_HOST_DEVICE(int, int) {});
  auto snd2 = ex::just() //
            | ex::bulk_chunked(ex::par, 9, [] _CCCL_HOST_DEVICE(int, int) {});
  auto snd3 = ex::just() //
            | ex::bulk_chunked(ex::par_unseq, 9, [] _CCCL_HOST_DEVICE(int, int) {});
  auto snd4 = ex::just() //
            | ex::bulk_chunked(ex::unseq, 9, [] _CCCL_HOST_DEVICE(int, int) {});

  STATIC_REQUIRE(ex::sender<decltype(snd1)>);
  STATIC_REQUIRE(ex::sender<decltype(snd2)>);
  STATIC_REQUIRE(ex::sender<decltype(snd3)>);
  STATIC_REQUIRE(ex::sender<decltype(snd4)>);
  (void) snd1;
  (void) snd2;
  (void) snd3;
  (void) snd4;
}

TEST_CASE("bulk forwards values", "[adaptors][bulk]")
{
  constexpr int n            = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto sndr = ex::just(magic_number, &counter) //
            | ex::bulk(ex::par, n, [](int i, int val, int(*counter)[n]) {
                if (val == magic_number)
                {
                  (*counter)[i]++;
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number, &counter});
  ex::start(op);

  for (int i : counter)
  {
    CHECK(i == 1);
  }
}

TEST_CASE("bulk_chunked forwards values", "[adaptors][bulk]")
{
  constexpr int n            = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto sndr = ex::just(magic_number) //
            | ex::bulk_chunked(ex::par, n, [&](int b, int e, int val) {
                if (val == magic_number)
                {
                  while (b < e)
                  {
                    counter[b++]++;
                  }
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

TEST_CASE("bulk_unchunked forwards values", "[adaptors][bulk]")
{
  constexpr int n            = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto sndr = ex::just(magic_number) //
            | ex::bulk_unchunked(n, [&](int i, int val) {
                if (val == magic_number)
                {
                  counter[i]++;
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

constexpr std::size_t n = 9;

TEST_CASE("bulk forwards values that can be taken by reference", "[adaptors][bulk]")
{
  _CUDA_VSTD::array<int, n> vals{};
  _CUDA_VSTD::array<int, n> vals_expected{};
  _CUDA_VSTD::iota(vals_expected.begin(), vals_expected.end(), 0);

  auto sndr = ex::just(cuda::std::move(vals)) //
            | ex::bulk(ex::par, n, [&](std::size_t i, _CUDA_VSTD::array<int, n>& vals) {
                vals[i] = static_cast<int>(i);
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{vals_expected});
  ex::start(op);
}

TEST_CASE("bulk_chunked forwards values that can be taken by reference", "[adaptors][bulk]")
{
  _CUDA_VSTD::array<int, n> vals{};
  _CUDA_VSTD::array<int, n> vals_expected{};
  _CUDA_VSTD::iota(vals_expected.begin(), vals_expected.end(), 0);

  auto sndr = ex::just(cuda::std::move(vals)) //
            | ex::bulk_chunked(ex::par, n, [&](std::size_t b, std::size_t e, _CUDA_VSTD::array<int, n>& vals) {
                for (; b != e; ++b)
                {
                  vals[b] = static_cast<int>(b);
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{vals_expected});
  ex::start(op);
}

TEST_CASE("bulk_unchunked forwards values that can be taken by reference", "[adaptors][bulk]")
{
  _CUDA_VSTD::array<int, n> vals{};
  _CUDA_VSTD::array<int, n> vals_expected{};
  _CUDA_VSTD::iota(vals_expected.begin(), vals_expected.end(), 0);

  auto sndr = ex::just(cuda::std::move(vals)) //
            | ex::bulk_unchunked(n, [&](std::size_t i, _CUDA_VSTD::array<int, n>& vals) {
                vals[i] = static_cast<int>(i);
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{vals_expected});
  ex::start(op);
}

TEST_CASE("bulk cannot be used to change the value type", "[adaptors][bulk]")
{
  constexpr int magic_number = 42;
  constexpr int n            = 2;

  auto sndr = ex::just(magic_number) //
            | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {
                return function_object_t<int>{nullptr};
              });

  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);
}

TEST_CASE("bulk_chunked cannot be used to change the value type", "[adaptors][bulk]")
{
  constexpr int magic_number = 42;
  constexpr int n            = 2;

  auto sndr = ex::just(magic_number) //
            | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int, int) {
                return function_object_range_t<int>{nullptr};
              });

  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);
}

TEST_CASE("bulk_unchunked cannot be used to change the value type", "[adaptors][bulk]")
{
  constexpr int magic_number = 42;
  constexpr int n            = 2;

  auto sndr = ex::just(magic_number) //
            | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int, int) {
                return function_object_t<int>{nullptr};
              });

  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);
}

#if _CCCL_HAS_EXCEPTIONS() && !defined(__CUDA_ARCH__)
TEST_CASE("bulk can throw, and set_error will be called", "[adaptors][bulk]")
{
  constexpr int n = 2;

  auto sndr = ex::just() //
            | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) -> int {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{});
  ex::start(op);
}

TEST_CASE("bulk_chunked can throw, and set_error will be called", "[adaptors][bulk]")
{
  constexpr int n = 2;

  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) -> int {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{});
  ex::start(op);
}

TEST_CASE("bulk_unchunked can throw, and set_error will be called", "[adaptors][bulk]")
{
  constexpr int n = 2;

  auto sndr = ex::just() //
            | ex::bulk_unchunked(n, [] _CCCL_HOST_DEVICE(int) -> int {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{});
  ex::start(op);
}
#endif // _CCCL_HAS_EXCEPTIONS() && !defined(__CUDA_ARCH__)

TEST_CASE("bulk function is not called on error", "[adaptors][bulk]")
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_error(string{"err"}) //
            | ex::bulk(ex::par, n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{string{"err"}});
  ex::start(op);
}

TEST_CASE("bulk_chunked function is not called on error", "[adaptors][bulk]")
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_error(string{"err"}) //
            | ex::bulk_chunked(ex::par, n, [&called](int, int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{string{"err"}});
  ex::start(op);
}

TEST_CASE("bulk_unchunked function is not called on error", "[adaptors][bulk]")
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_error(string{"err"}) //
            | ex::bulk_unchunked(n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{string{"err"}});
  ex::start(op);
}

TEST_CASE("bulk function in not called on stop", "[adaptors][bulk]")
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_stopped() //
            | ex::bulk(ex::par, n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("bulk_chunked function in not called on stop", "[adaptors][bulk]")
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_stopped() //
            | ex::bulk_chunked(ex::par, n, [&called](int, int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("bulk_unchunked function in not called on stop", "[adaptors][bulk]")
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_stopped() //
            | ex::bulk_unchunked(n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("default bulk works with non-default constructible types", "[adaptors][bulk]")
{
  auto s = ex::just(non_default_constructible{42}) //
         | ex::bulk(ex::par, 1, [] _CCCL_HOST_DEVICE(int, ignore_lvalue_ref) {});
  ex::sync_wait(cuda::std::move(s));
}

TEST_CASE("default bulk_chunked works with non-default constructible types", "[adaptors][bulk]")
{
  auto s = ex::just(non_default_constructible{42}) //
         | ex::bulk_chunked(ex::par, 1, [] _CCCL_HOST_DEVICE(int, int, ignore_lvalue_ref) {});
  ex::sync_wait(cuda::std::move(s));
}

TEST_CASE("default bulk_unchunked works with non-default constructible types", "[adaptors][bulk]")
{
  auto s = ex::just(non_default_constructible{42}) //
         | ex::bulk_unchunked(1, [] _CCCL_HOST_DEVICE(int, ignore_lvalue_ref) {});
  ex::sync_wait(cuda::std::move(s));
}

#if !defined(__CUDA_ARCH__)
// TODO: modify these tests to work on device as well
struct my_domain
{
  _CCCL_TEMPLATE(class Sender, class... Env)
  _CCCL_REQUIRES(ex::sender_for<Sender, ex::bulk_chunked_t>)
  static auto transform_sender(Sender, const Env&...)
  {
    return ex::just(string{"hijacked"});
  }
};

TEST_CASE("late customizing bulk_chunked also changes the behavior of bulk", "[adaptors][then]")
{
  bool called{false};
  // The customization will return a different value
  inline_scheduler<my_domain> sched;
  auto sndr = ex::just(string{"hello"}) //
            | ex::continues_on(sched) //
            | ex::bulk(ex::par, 1, [&called](int, string) {
                called = true;
              });
  wait_for_value(cuda::std::move(sndr), string{"hijacked"});
  REQUIRE_FALSE(called);
}

struct my_domain2
{
  _CCCL_TEMPLATE(class Sender, class... Env)
  _CCCL_REQUIRES(ex::sender_for<Sender, ex::bulk_t>)
  static auto transform_sender(Sender, const Env&...)
  {
    return ex::just(string{"hijacked"});
  }
};

TEST_CASE("bulk can be customized, independently of bulk_chunked", "[adaptors][then]")
{
  bool called{false};
  // The customization will return a different value
  inline_scheduler<my_domain2> sched;
  auto sndr = ex::just(string{"hello"}) //
            | ex::continues_on(sched) //
            | ex::bulk(ex::par, 1, [&called](int, string) {
                called = true;
              });
  wait_for_value(cuda::std::move(sndr), string{"hijacked"});
  REQUIRE_FALSE(called);

  // bulk_chunked will still use the default implementation
  auto snd2 = ex::just(string{"hello"}) //
            | ex::continues_on(sched) | ex::bulk_chunked(ex::par, 1, [&called](int, int, string) {
                called = true;
              });
  wait_for_value(cuda::std::move(snd2), string{"hello"});
  REQUIRE(called);
}
#endif // !defined(__CUDA_ARCH__)

} // namespace
