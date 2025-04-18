//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

//  constexpr span(const span& other) noexcept = default;

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool doCopy(const T& rhs)
{
  static_assert(noexcept(T{rhs}));
  T lhs{rhs};
  return lhs.data() == rhs.data() && lhs.size() == rhs.size();
}

struct A
{};

template <typename T>
__host__ __device__ void testCV()
{
  int arr[] = {1, 2, 3};
  assert((doCopy(cuda::std::span<T>())));
  assert((doCopy(cuda::std::span<T, 0>())));
  assert((doCopy(cuda::std::span<T>(&arr[0], 1))));
  assert((doCopy(cuda::std::span<T, 1>(&arr[0], 1))));
  assert((doCopy(cuda::std::span<T>(&arr[0], 2))));
  assert((doCopy(cuda::std::span<T, 2>(&arr[0], 2))));
}

TEST_GLOBAL_VARIABLE constexpr int carr[] = {1, 2, 3};

int main(int, char**)
{
  static_assert(doCopy(cuda::std::span<int>()));
  static_assert(doCopy(cuda::std::span<int, 0>()));
  static_assert(doCopy(cuda::std::span<const int>(&carr[0], 1)));
  static_assert(doCopy(cuda::std::span<const int, 1>(&carr[0], 1)));
  static_assert(doCopy(cuda::std::span<const int>(&carr[0], 2)));
  static_assert(doCopy(cuda::std::span<const int, 2>(&carr[0], 2)));

  static_assert(doCopy(cuda::std::span<long>()));
  static_assert(doCopy(cuda::std::span<double>()));
  static_assert(doCopy(cuda::std::span<A>()));

  testCV<int>();
  testCV<const int>();
  testCV<volatile int>();
  testCV<const volatile int>();

  return 0;
}
