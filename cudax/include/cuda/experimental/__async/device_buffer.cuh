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

_CCCL_NV_DIAG_SUPPRESS(20011)

#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <functional> // for function
#include <memory> // for unique_ptr

namespace cuda::experimental::__async
{

struct device_memory_resource;

struct device_buffer_deleter
{
  template <class _Fn>
  device_buffer_deleter(_Fn __fn, stream_ref __stream)
      : __stream_(__stream)
      , __fn_(cuda::std::move(__fn))
  {}

  void operator()(const void* __pv) const noexcept
  {
    __fn_(__pv, __stream_);
  }

  stream_ref __stream_;
  ::std::function<void(const void*, stream_ref)> __fn_;
};

template <class _Type>
struct device_buffer : ::std::unique_ptr<_Type[], device_buffer_deleter>
{
  using value_type = _Type;

  size_t size() const noexcept
  {
    return __count_;
  }

  cuda::stream_ref get_stream() const
  {
    return this->get_deleter().__stream_;
  }

private:
  friend device_memory_resource;

  friend auto __cudax_launch_transform(stream_ref stream, device_buffer& self) noexcept
  {
    return cuda::std::span<_Type>{self.get(), self.size()};
  }

  friend auto __cudax_launch_transform(stream_ref stream, const device_buffer& self) noexcept
  {
    return cuda::std::span<const _Type>{self.get(), self.size()};
  }

  template <class _Deleter>
  device_buffer(_Type* __ptr, size_t __count, stream_ref __stream, _Deleter __fn)
      : device_buffer::unique_ptr(__ptr, device_buffer_deleter(cuda::std::move(__fn), __stream))
      , __count_(__count)
  {}

  size_t __count_;
};

} // namespace cuda::experimental::__async

#endif // __CUDAX_ASYNC_DETAIL_DEVICE_BUFFER
