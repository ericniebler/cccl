//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   log(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
  assert(log(c) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(0, 0), cuda::std::complex<T>(-cuda::std::numeric_limits<T>::infinity(), 0));
}

template <class T>
__host__ __device__ void test_edges()
{
  const T pi       = cuda::std::atan2(+0., -0.);
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    cuda::std::complex<T> r = log(testcases[i]);
    if (testcases[i].real() == T(0) && testcases[i].imag() == T(0))
    {
      if (cuda::std::signbit(testcases[i].real()))
      {
        assert(cuda::std::isinf(r.real()));
        assert(r.real() < T(0));
        if (cuda::std::signbit(testcases[i].imag()))
        {
          is_about(r.imag(), -pi);
        }
        else
        {
          is_about(r.imag(), pi);
        }
      }
      else
      {
        assert(cuda::std::isinf(r.real()));
        assert(r.real() < T(0));
        assert(r.imag() == T(0));
        assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
      }
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
    {
      assert(cuda::std::isinf(r.real()));
      assert(r.real() > T(0));
      if (testcases[i].imag() > T(0))
      {
        is_about(r.imag(), pi / T(2));
      }
      else
      {
        is_about(r.imag(), -pi / T(2));
      }
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0)
             && cuda::std::isfinite(testcases[i].imag()))
    {
      assert(cuda::std::isinf(r.real()) && r.real() > T(0));
      if (r.imag() > T(0))
      {
        is_about(r.imag(), pi);
      }
      else
      {
        is_about(r.imag(), -pi);
      }
    }
    else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0)
             && cuda::std::isfinite(testcases[i].imag()))
    {
      assert(cuda::std::isinf(r.real()) && r.real() > T(0));
      assert(r.imag() == T(0));
      assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
    }
    else if (testcases[i].real() == T(1) && testcases[i].imag() == T(0))
    {
      assert(r.real() == T(0));
      assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
    }
    else if (testcases[i].real() == T(0) && testcases[i].imag() == T(1))
    {
      assert(r.real() == T(0));
      is_about(r.imag(), pi / T(2));
    }
    else if (testcases[i].real() == -T(1) && testcases[i].imag() == T(0))
    {
      assert(r.real() == T(0));
      if (cuda::std::signbit(testcases[i].imag()))
      {
        is_about(r.imag(), -pi);
      }
      else
      {
        is_about(r.imag(), pi);
      }
    }
    else if (testcases[i].real() == T(0) && testcases[i].imag() == -T(1))
    {
      assert(r.real() == T(0));
      is_about(r.imag(), -pi / T(2));
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag())
             && abs(testcases[i]) < T(1))
    {
      assert(cuda::std::signbit(r.real()));
      assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag())
             && abs(testcases[i]) > T(1))
    {
      assert(!cuda::std::signbit(r.real()));
      assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_edges<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
