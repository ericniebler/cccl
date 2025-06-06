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
#include <thrust/advance.h>
#include <thrust/count.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/system/detail/generic/partition.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator stable_partition(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  // copy input to temp buffer
  thrust::detail::temporary_array<InputType, DerivedPolicy> temp(exec, first, last);

  // count the size of the true partition
  thrust::detail::it_difference_t<ForwardIterator> num_true = thrust::count_if(exec, first, last, pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  ::cuda::std::advance(out_false, num_true);

  return thrust::stable_partition_copy(exec, temp.begin(), temp.end(), first, out_false, pred).first;
} // end stable_partition()

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator stable_partition(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  // copy input to temp buffer
  thrust::detail::temporary_array<InputType, DerivedPolicy> temp(exec, first, last);

  // count the size of the true partition
  InputIterator stencil_last = stencil;
  ::cuda::std::advance(stencil_last, temp.size());
  thrust::detail::it_difference_t<InputIterator> num_true = thrust::count_if(exec, stencil, stencil_last, pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  ::cuda::std::advance(out_false, num_true);

  return thrust::stable_partition_copy(exec, temp.begin(), temp.end(), stencil, first, out_false, pred).first;
} // end stable_partition()

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE thrust::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  auto not_pred = ::cuda::std::not_fn(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = thrust::remove_copy_if(exec, first, last, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = thrust::remove_copy_if(exec, first, last, out_false, pred);

  return thrust::make_pair(end_of_true_partition, end_of_false_partition);
} // end stable_partition_copy()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE thrust::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  auto not_pred = ::cuda::std::not_fn(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = thrust::remove_copy_if(exec, first, last, stencil, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = thrust::remove_copy_if(exec, first, last, stencil, out_false, pred);

  return thrust::make_pair(end_of_true_partition, end_of_false_partition);
} // end stable_partition_copy()

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator
partition(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  return thrust::stable_partition(exec, first, last, pred);
} // end partition()

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator partition(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  return thrust::stable_partition(exec, first, last, stencil, pred);
} // end partition()

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE thrust::pair<OutputIterator1, OutputIterator2> partition_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  return thrust::stable_partition_copy(exec, first, last, out_true, out_false, pred);
} // end partition_copy()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
_CCCL_HOST_DEVICE thrust::pair<OutputIterator1, OutputIterator2> partition_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  return thrust::stable_partition_copy(exec, first, last, stencil, out_true, out_false, pred);
} // end partition_copy()

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator partition_point(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  return thrust::find_if_not(exec, first, last, pred);
} // end partition_point()

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
is_partitioned(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::is_sorted(exec,
                           thrust::make_transform_iterator(first, ::cuda::std::not_fn(pred)),
                           thrust::make_transform_iterator(last, ::cuda::std::not_fn(pred)));
} // end is_partitioned()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
