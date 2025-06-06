//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATOMIC_HELPERS_H
#define ATOMIC_HELPERS_H

#include <cuda/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

struct UserAtomicType
{
  int i;

  __host__ __device__ explicit UserAtomicType(int d = 0) noexcept
      : i(d)
  {}

  __host__ __device__ friend bool operator==(const UserAtomicType& x, const UserAtomicType& y)
  {
    return x.i == y.i;
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          >
struct TestEachIntegralType
{
  __host__ __device__ void operator()() const
  {
    TestFunctor<char, Selector, Scope>()();
    TestFunctor<signed char, Selector, Scope>()();
    TestFunctor<unsigned char, Selector, Scope>()();
    TestFunctor<short, Selector, Scope>()();
    TestFunctor<unsigned short, Selector, Scope>()();
    TestFunctor<int, Selector, Scope>()();
    TestFunctor<unsigned int, Selector, Scope>()();
    TestFunctor<long, Selector, Scope>()();
    TestFunctor<unsigned long, Selector, Scope>()();
    TestFunctor<long long, Selector, Scope>()();
    TestFunctor<unsigned long long, Selector, Scope>()();
    TestFunctor<wchar_t, Selector, Scope>();
    TestFunctor<char16_t, Selector, Scope>()();
    TestFunctor<char32_t, Selector, Scope>()();
    TestFunctor<int8_t, Selector, Scope>()();
    TestFunctor<uint8_t, Selector, Scope>()();
    TestFunctor<int16_t, Selector, Scope>()();
    TestFunctor<uint16_t, Selector, Scope>()();
    TestFunctor<int32_t, Selector, Scope>()();
    TestFunctor<uint32_t, Selector, Scope>()();
    TestFunctor<int64_t, Selector, Scope>()();
    TestFunctor<uint64_t, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          >
struct TestEachFloatingPointType
{
  __host__ __device__ void operator()() const
  {
    TestFunctor<float, Selector, Scope>()();
    TestFunctor<double, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          >
struct TestEachAtomicType
{
  __host__ __device__ void operator()() const
  {
    TestEachIntegralType<TestFunctor, Selector, Scope>()();
    TestEachFloatingPointType<TestFunctor, Selector, Scope>()();
    TestFunctor<UserAtomicType, Selector, Scope>()();
    TestFunctor<int*, Selector, Scope>()();
    TestFunctor<const int*, Selector, Scope>()();
  }
};

#endif // ATOMIC_HELPER_H
