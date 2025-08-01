//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_default_constructible

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_default_constructible()
{
  static_assert(cuda::std::is_default_constructible<T>::value, "");
  static_assert(cuda::std::is_default_constructible<const T>::value, "");
  static_assert(cuda::std::is_default_constructible<volatile T>::value, "");
  static_assert(cuda::std::is_default_constructible<const volatile T>::value, "");
  static_assert(cuda::std::is_default_constructible_v<T>, "");
  static_assert(cuda::std::is_default_constructible_v<const T>, "");
  static_assert(cuda::std::is_default_constructible_v<volatile T>, "");
  static_assert(cuda::std::is_default_constructible_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_default_constructible()
{
  static_assert(!cuda::std::is_default_constructible<T>::value, "");
  static_assert(!cuda::std::is_default_constructible<const T>::value, "");
  static_assert(!cuda::std::is_default_constructible<volatile T>::value, "");
  static_assert(!cuda::std::is_default_constructible<const volatile T>::value, "");
  static_assert(!cuda::std::is_default_constructible_v<T>, "");
  static_assert(!cuda::std::is_default_constructible_v<const T>, "");
  static_assert(!cuda::std::is_default_constructible_v<volatile T>, "");
  static_assert(!cuda::std::is_default_constructible_v<const volatile T>, "");
}

class Empty
{};

class NoDefaultConstructor
{
  __host__ __device__ NoDefaultConstructor(int) {}
};

class NotEmpty
{
public:
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
public:
  __host__ __device__ virtual ~Abstract() = 0;
};

struct A
{
  __host__ __device__ A();
};

class B
{
  __host__ __device__ B();
};

int main(int, char**)
{
  test_is_default_constructible<A>();
  test_is_default_constructible<Union>();
  test_is_default_constructible<Empty>();
  test_is_default_constructible<int>();
  test_is_default_constructible<double>();
  test_is_default_constructible<int*>();
  test_is_default_constructible<const int*>();
  test_is_default_constructible<char[3]>();
  test_is_default_constructible<char[5][3]>();

  test_is_default_constructible<NotEmpty>();
  test_is_default_constructible<bit_zero>();

  test_is_not_default_constructible<void>();
  test_is_not_default_constructible<int&>();
  test_is_not_default_constructible<char[]>();
  test_is_not_default_constructible<char[][3]>();

  test_is_not_default_constructible<Abstract>();
  test_is_not_default_constructible<NoDefaultConstructor>();
  test_is_not_default_constructible<B>();
  test_is_not_default_constructible<int&&>();

  test_is_not_default_constructible<void()>();
  test_is_not_default_constructible<void() const>();
  test_is_not_default_constructible<void() volatile>();
  test_is_not_default_constructible<void() &>();
  test_is_not_default_constructible<void() &&>();

  return 0;
}
