//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: libcpp-no-structured-bindings
// UNSUPPORTED: msvc

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct S
{
  int x;
};

__host__ __device__ void test_decomp_user_type()
{
  {
    S s{99};
    auto [m1]  = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &s.x);
  }
  {
    S const s{99};
    auto [m1]  = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &s.x);
  }
}

__host__ __device__ void test_decomp_tuple()
{
  using T = cuda::std::tuple<int>;
  {
    T s{99};
    auto [m1]  = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &cuda::std::get<0>(s));
  }
  {
    T const s{99};
    auto [m1]  = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &cuda::std::get<0>(s));
  }
}

__host__ __device__ void test_decomp_pair()
{
  using T = cuda::std::pair<int, double>;
  {
    T s{99, 42.5};
    auto [m1, m2]  = s;
    auto& [r1, r2] = s;
    assert(m1 == 99);
    assert(m2 == 42.5);
    assert(&r1 == &cuda::std::get<0>(s));
    assert(&r2 == &cuda::std::get<1>(s));
  }
  {
    T const s{99, 42.5};
    auto [m1, m2]  = s;
    auto& [r1, r2] = s;
    assert(m1 == 99);
    assert(m2 == 42.5);
    assert(&r1 == &cuda::std::get<0>(s));
    assert(&r2 == &cuda::std::get<1>(s));
  }
}

__host__ __device__ void test_decomp_array()
{
  using T = cuda::std::array<int, 3>;
  {
    T s{{99, 42, -1}};
    auto [m1, m2, m3]  = s;
    auto& [r1, r2, r3] = s;
    assert(m1 == 99);
    assert(m2 == 42);
    assert(m3 == -1);
    assert(&r1 == &cuda::std::get<0>(s));
    assert(&r2 == &cuda::std::get<1>(s));
    assert(&r3 == &cuda::std::get<2>(s));
  }
  {
    T const s{{99, 42, -1}};
    auto [m1, m2, m3]  = s;
    auto& [r1, r2, r3] = s;
    assert(m1 == 99);
    assert(m2 == 42);
    assert(m3 == -1);
    assert(&r1 == &cuda::std::get<0>(s));
    assert(&r2 == &cuda::std::get<1>(s));
    assert(&r3 == &cuda::std::get<2>(s));
  }
}

struct Test
{
  int x;
};

template <size_t N>
__host__ __device__ int get(Test const&)
{
  static_assert(N == 0, "");
  return -1;
}

template <>
struct std::tuple_element<0, Test>
{
  using type = int;
};

__host__ __device__ void test_before_tuple_size_specialization()
{
  Test const t{99};
  auto& [p] = t;
  assert(p == 99);
}

template <>
struct std::tuple_size<Test>
{
public:
  static const size_t value = 1;
};

__host__ __device__ void test_after_tuple_size_specialization()
{
  Test const t{99};
  auto& [p] = t;
#if !_CCCL_COMPILER(NVRTC) // nvbug4053842
  assert(p == -1);
#endif
}

int main(int, char**)
{
  test_decomp_user_type();
  test_decomp_tuple();
  test_decomp_pair();
  test_decomp_array();
  test_before_tuple_size_specialization();
  test_after_tuple_size_specialization();

  return 0;
}
