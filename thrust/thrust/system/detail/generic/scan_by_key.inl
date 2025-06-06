/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/scan_by_key.h>
#include <thrust/transform.h>

#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

template <typename OutputType, typename HeadFlagType, typename AssociativeOperator>
struct segmented_scan_functor
{
  AssociativeOperator binary_op;

  using result_type = typename thrust::tuple<OutputType, HeadFlagType>;

  _CCCL_HOST_DEVICE segmented_scan_functor(AssociativeOperator _binary_op)
      : binary_op(_binary_op)
  {}

  _CCCL_HOST_DEVICE result_type operator()(result_type a, result_type b)
  {
    return result_type(thrust::get<1>(b) ? thrust::get<0>(b) : binary_op(thrust::get<0>(a), thrust::get<0>(b)),
                       thrust::get<1>(a) | thrust::get<1>(b));
  }
};

} // end namespace detail

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result)
{
  return thrust::inclusive_scan_by_key(exec, first1, last1, first2, result, ::cuda::std::equal_to<>());
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred)
{
  return thrust::inclusive_scan_by_key(exec, first1, last1, first2, result, binary_pred, ::cuda::std::plus<>());
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  using OutputType   = thrust::detail::it_value_t<InputIterator2>;
  using HeadFlagType = std::uint8_t;

  const size_t n = last1 - first1;

  if (n != 0)
  {
    // compute head flags
    thrust::detail::temporary_array<HeadFlagType, DerivedPolicy> flags(exec, n);
    flags[0] = 1;
    thrust::transform(exec, first1, last1 - 1, first1 + 1, flags.begin() + 1, ::cuda::std::not_fn(binary_pred));

    // scan key-flag tuples,
    // For additional details refer to Section 2 of the following paper
    //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
    //    NVIDIA Technical Report NVR-2008-003, December 2008
    //    http://mgarland.org/files/papers/nvr-2008-003.pdf
    thrust::inclusive_scan(
      exec,
      thrust::make_zip_iterator(first2, flags.begin()),
      thrust::make_zip_iterator(first2, flags.begin()) + n,
      thrust::make_zip_iterator(result, flags.begin()),
      detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
  }

  return result + n;
}

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result)
{
  using InitType = thrust::detail::it_value_t<InputIterator2>;
  return thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, InitType{});
}

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init)
{
  return thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, init, ::cuda::std::equal_to<>());
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred)
{
  return thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, init, binary_pred, ::cuda::std::plus<>());
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  using OutputType   = T;
  using HeadFlagType = std::uint8_t;

  const size_t n = last1 - first1;

  if (n != 0)
  {
    InputIterator2 last2 = first2 + n;

    // compute head flags
    thrust::detail::temporary_array<HeadFlagType, DerivedPolicy> flags(exec, n);
    flags[0] = 1;
    thrust::transform(exec, first1, last1 - 1, first1 + 1, flags.begin() + 1, ::cuda::std::not_fn(binary_pred));

    // shift input one to the right and initialize segments with init
    thrust::detail::temporary_array<OutputType, DerivedPolicy> temp(exec, n);
    thrust::replace_copy_if(
      exec, first2, last2 - 1, flags.begin() + 1, temp.begin() + 1, ::cuda::std::negate<HeadFlagType>(), init);
    temp[0] = init;

    // scan key-flag tuples,
    // For additional details refer to Section 2 of the following paper
    //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
    //    NVIDIA Technical Report NVR-2008-003, December 2008
    //    http://mgarland.org/files/papers/nvr-2008-003.pdf
    thrust::inclusive_scan(
      exec,
      thrust::make_zip_iterator(temp.begin(), flags.begin()),
      thrust::make_zip_iterator(temp.begin(), flags.begin()) + n,
      thrust::make_zip_iterator(result, flags.begin()),
      detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
  }

  return result + n;
}

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
