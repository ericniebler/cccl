//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class Iter1, class Iter2>
//   bool operator<=(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

// move_iterator's operator<= calls the underlying iterator's operator<=
struct CustomIt
{
  using value_type        = int;
  using difference_type   = int;
  using reference         = int&;
  using pointer           = int*;
  using iterator_category = cuda::std::input_iterator_tag;
  CustomIt()              = default;
  __host__ __device__ constexpr explicit CustomIt(int* p)
      : p_(p)
  {}
  __host__ __device__ int& operator*() const;
  __host__ __device__ CustomIt& operator++();
  __host__ __device__ CustomIt operator++(int);
  __host__ __device__ constexpr friend bool operator<=(const CustomIt& a, const CustomIt& b)
  {
    return a.p_ <= b.p_;
  }
  int* p_ = nullptr;
};

template <class It>
__host__ __device__ constexpr void test_one()
{
  int a[]                               = {3, 1, 4};
  const cuda::std::move_iterator<It> r1 = cuda::std::move_iterator<It>(It(a));
  const cuda::std::move_iterator<It> r2 = cuda::std::move_iterator<It>(It(a + 2));
  static_assert(cuda::std::is_same_v<decltype(r1 <= r2), bool>);
  assert((r1 <= r1));
  assert((r1 <= r2));
  assert(!(r2 <= r1));
}

__host__ __device__ constexpr bool test()
{
  test_one<CustomIt>();
  test_one<int*>();
  test_one<const int*>();
  test_one<random_access_iterator<int*>>();
  test_one<contiguous_iterator<int*>>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test(), "");

  return 0;
}
