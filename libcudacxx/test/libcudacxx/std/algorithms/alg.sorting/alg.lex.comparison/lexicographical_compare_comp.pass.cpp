//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2, CopyConstructible Compare>
//   requires Predicate<Compare, Iter1::value_type, Iter2::value_type>
//         && Predicate<Compare, Iter2::value_type, Iter1::value_type>
//   constexpr bool             // constexpr after C++17
//   lexicographical_compare(Iter1 first1, Iter1 last1,
//                           Iter2 first2, Iter2 last2, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class Iter2>
__host__ __device__ constexpr void test()
{
  int ia[]          = {1, 2, 3, 4};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ib[]          = {1, 2, 3};
  typedef cuda::std::greater<int> C;
  C c{};
  assert(!cuda::std::lexicographical_compare(Iter1(ia), Iter1(ia + sa), Iter2(ib), Iter2(ib + 2), c));
  assert(cuda::std::lexicographical_compare(Iter1(ib), Iter1(ib + 2), Iter2(ia), Iter2(ia + sa), c));
  assert(!cuda::std::lexicographical_compare(Iter1(ia), Iter1(ia + sa), Iter2(ib), Iter2(ib + 3), c));
  assert(cuda::std::lexicographical_compare(Iter1(ib), Iter1(ib + 3), Iter2(ia), Iter2(ia + sa), c));
  assert(!cuda::std::lexicographical_compare(Iter1(ia), Iter1(ia + sa), Iter2(ib + 1), Iter2(ib + 3), c));
  assert(cuda::std::lexicographical_compare(Iter1(ib + 1), Iter1(ib + 3), Iter2(ia), Iter2(ia + sa), c));
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, const int*>();

  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>>();
  test<forward_iterator<const int*>, const int*>();

  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, const int*>();

  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>>();
  test<random_access_iterator<const int*>, const int*>();

  test<const int*, cpp17_input_iterator<const int*>>();
  test<const int*, forward_iterator<const int*>>();
  test<const int*, bidirectional_iterator<const int*>>();
  test<const int*, random_access_iterator<const int*>>();
  test<const int*, const int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
