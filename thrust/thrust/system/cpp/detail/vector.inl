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
#include <thrust/system/cpp/vector.h>

#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cpp
{

template <typename T, typename Allocator>
vector<T, Allocator>::vector()
    : super_t()
{}

template <typename T, typename Allocator>
vector<T, Allocator>::vector(size_type n)
    : super_t(n)
{}

template <typename T, typename Allocator>
vector<T, Allocator>::vector(size_type n, const value_type& value)
    : super_t(n, value)
{}

template <typename T, typename Allocator>
vector<T, Allocator>::vector(const vector& x)
    : super_t(x)
{}

template <typename T, typename Allocator>
vector<T, Allocator>::vector(vector&& x)
    : super_t(::cuda::std::move(x))
{}

template <typename T, typename Allocator>
template <typename OtherT, typename OtherAllocator>
vector<T, Allocator>::vector(const thrust::detail::vector_base<OtherT, OtherAllocator>& x)
    : super_t(x)
{}

template <typename T, typename Allocator>
template <typename OtherT, typename OtherAllocator>
vector<T, Allocator>::vector(const std::vector<OtherT, OtherAllocator>& x)
    : super_t(x)
{}

template <typename T, typename Allocator>
template <typename InputIterator>
vector<T, Allocator>::vector(InputIterator first, InputIterator last)
    : super_t(first, last)
{}

template <typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(const vector& x)
{
  super_t::operator=(x);
  return *this;
}

template <typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(vector&& x)
{
  super_t::operator=(::cuda::std::move(x));
  return *this;
}

template <typename T, typename Allocator>
vector<T, Allocator>::vector(::cuda::std::initializer_list<T> il)
    : super_t(il)
{}

template <typename T, typename Allocator>
vector<T, Allocator>::vector(::cuda::std::initializer_list<T> il, const Allocator& alloc)
    : super_t(il, alloc)
{}

template <typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(::cuda::std::initializer_list<T> il)
{
  super_t::operator=(il);
  return *this;
}

template <typename T, typename Allocator>
template <typename OtherT, typename OtherAllocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(const std::vector<OtherT, OtherAllocator>& x)
{
  super_t::operator=(x);
  return *this;
}

template <typename T, typename Allocator>
template <typename OtherT, typename OtherAllocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(const thrust::detail::vector_base<OtherT, OtherAllocator>& x)
{
  super_t::operator=(x);
  return *this;
}

} // namespace cpp
} // namespace system
THRUST_NAMESPACE_END
