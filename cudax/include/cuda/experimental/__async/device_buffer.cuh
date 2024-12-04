//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_DEVICE_BUFFER
#define __CUDAX_ASYNC_DETAIL_DEVICE_BUFFER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__stream/stream_ref.cuh>

#include <functional> // for function
#include <memory> // for unique_ptr

namespace cuda::experimental::async
{

struct device_memory;

struct device_buffer_deleter
{
  template <class _Fn>
  device_buffer_deleter(_Fn __fn, stream_ref __stream)
      : __stream_(__stream)
      , __fn_(_CUDA_VSTD::move(__fn))
  {}

  void operator()(const void* __pv) const noexcept
  {
    __fn_(__pv, __stream_);
  }

  stream_ref __stream_;
  ::std::function<void(const void*, stream_ref)> __fn_;
};

struct __immutable_buffer_base
{
  _CUDAX_TRIVIAL_HOST_API size_t size() const noexcept
  {
    return __count_;
  }

protected:
  _CUDAX_TRIVIAL_HOST_API explicit __immutable_buffer_base(size_t __count) noexcept
      : __count_(__count)
  {}

  size_t __count_;
};

template <class _Type>
struct device_buffer : private __immutable_buffer_base
{
  using value_type = _Type;

  using __immutable_buffer_base::size;

  _CUDAX_TRIVIAL_HOST_API auto get() noexcept -> value_type*
  {
    return __array_.get();
  }

  _CUDAX_TRIVIAL_HOST_API auto get() const noexcept -> value_type const*
  {
    return __array_.get();
  }

  _CUDAX_TRIVIAL_HOST_API stream_ref get_stream() const
  {
    return this->get_deleter().__stream_;
  }

  _CUDAX_TRIVIAL_HOST_API auto data() noexcept -> value_type*
  {
    return __array_.get();
  }

  _CUDAX_TRIVIAL_HOST_API auto data() const noexcept -> const value_type*
  {
    return __array_.get();
  }

  _CUDAX_TRIVIAL_HOST_API auto begin() noexcept -> value_type*
  {
    return __array_.get();
  }

  _CUDAX_TRIVIAL_HOST_API auto begin() const noexcept -> const value_type*
  {
    return __array_.get();
  }

  _CUDAX_TRIVIAL_HOST_API auto end() noexcept -> value_type*
  {
    return __array_.get() + size();
  }

  _CUDAX_TRIVIAL_HOST_API auto end() const noexcept -> const value_type*
  {
    return __array_.get() + size();
  }

  _CUDAX_TRIVIAL_DEVICE_API auto operator[](size_t __i) noexcept -> value_type&
  {
    return __array_[__i];
  }

  _CUDAX_TRIVIAL_DEVICE_API auto operator[](size_t __i) const noexcept -> const value_type&
  {
    return __array_[__i];
  }

  _CUDAX_TRIVIAL_HOST_API auto operator->() const noexcept -> __immutable_buffer_base const*
  {
    return this;
  }

private:
  friend device_memory;

  friend _CUDAX_HOST_API auto
  __cudax_launch_transform([[maybe_unused]] stream_ref __stream, device_buffer& __self) noexcept
  {
    return _CUDA_VSTD::span<_Type>{__self.data(), __self.size()};
  }

  friend _CUDAX_HOST_API auto
  __cudax_launch_transform([[maybe_unused]] stream_ref stream, const device_buffer& __self) noexcept
  {
    return _CUDA_VSTD::span<const _Type>{__self.data(), __self.size()};
  }

  template <class _Deleter>
  _CUDAX_HOST_API device_buffer(_Type* __ptr, size_t __count, stream_ref __stream, _Deleter __fn)
      : __immutable_buffer_base(__count)
      , __array_(__ptr, device_buffer_deleter(_CUDA_VSTD::move(__fn), __stream))
  {}

  ::std::unique_ptr<_Type[], device_buffer_deleter> __array_;
};

} // namespace cuda::experimental::async

#endif // __CUDAX_ASYNC_DETAIL_DEVICE_BUFFER
