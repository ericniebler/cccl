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

#include <cuda/experimental/__async/stream_ref.cuh>
#include <cuda/experimental/__async/transforms.cuh>
#include <cuda/experimental/__container/uninitialized_buffer.cuh>
#include <cuda/experimental/__memory_resource/device_memory_resource.cuh>

namespace cuda::experimental::async
{
struct device_memory;

struct __async_device_memory_resource
{
  _CUDAX_HOST_API explicit __async_device_memory_resource(stream_ref __stream) noexcept
      : __stream_(__stream)
  {}

  _CUDAX_PUBLIC_API void* allocate(size_t __bytes, [[maybe_unused]] size_t __align = alignof(max_align_t))
  {
    void* __pv = nullptr;
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync, //
      "Failed trying to asynchronously allocate device memory",
      &__pv,
      __bytes,
      __stream_.get());
    return __pv;
  }

  _CUDAX_PUBLIC_API void deallocate(void* __pv, //
                                    [[maybe_unused]] size_t __bytes,
                                    [[maybe_unused]] size_t __align = alignof(max_align_t)) noexcept
  {
    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync, //
      "Failed trying to asynchronously free device memory",
      __pv,
      __stream_.get());
  }

  _CUDAX_TRIVIAL_HOST_API stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

private:
  _CUDAX_PUBLIC_API _CCCL_NODISCARD_FRIEND bool
  operator==(const __async_device_memory_resource& __lhs, const __async_device_memory_resource& __rhs) noexcept
  {
    return __lhs.__stream_ == __rhs.__stream_;
  }

  _CUDAX_PUBLIC_API _CCCL_NODISCARD_FRIEND bool
  operator!=(const __async_device_memory_resource& __lhs, const __async_device_memory_resource& __rhs) noexcept
  {
    return __lhs.__stream_ != __rhs.__stream_;
  }

  stream_ref __stream_;
};

template <class _Buffer>
struct __immutable_buffer_base
{
  _CUDAX_HOST_API size_t size() const noexcept
  {
    using __base_t         = typename _Buffer::uninitialized_buffer;
    const __base_t& __base = static_cast<_Buffer const&>(*this);
    return __base.size();
  }
};

template <class _Type>
struct _CCCL_DECLSPEC_EMPTY_BASES device_buffer
    : uninitialized_buffer<_Type, device_accessible>
    , private __immutable_buffer_base<device_buffer<_Type>>
{
  using device_buffer::uninitialized_buffer::size;
  using typename device_buffer::uninitialized_buffer::value_type;

  device_buffer() = delete;

  _CUDAX_TRIVIAL_HOST_API auto operator->() const noexcept -> __immutable_buffer_base<device_buffer> const*
  {
    return this;
  }

private:
  friend device_memory;
  friend __immutable_buffer_base<device_buffer>;

  _CUDAX_HOST_API _CCCL_NODISCARD_FRIEND __stream_provider_transform<_CUDA_VSTD::span<_Type>>
  cuda_async_transform([[maybe_unused]] ::cuda::stream_ref __launch_stream, device_buffer& __self) noexcept
  {
    const auto& __mr         = __self.get_memory_resource();
    stream_ref __self_stream = __cudax::any_cast<__async_device_memory_resource>(&__mr)->get_stream();
    return __stream_provider_transform{
      _CUDA_VSTD::span<_Type>{__self.data(), __self.size()}, __self_stream, __launch_stream};
  }

  _CUDAX_HOST_API _CCCL_NODISCARD_FRIEND __stream_provider_transform<_CUDA_VSTD::span<_Type const>>
  cuda_async_transform([[maybe_unused]] ::cuda::stream_ref __launch_stream, const device_buffer& __self) noexcept
  {
    const auto& __mr         = __self.get_memory_resource();
    stream_ref __self_stream = __cudax::any_cast<__async_device_memory_resource>(&__mr)->get_stream();
    return __stream_provider_transform{
      _CUDA_VSTD::span<_Type const>{__self.data(), __self.size()}, __self_stream, __launch_stream};
  }

  _CUDAX_HOST_API device_buffer(size_t __count, stream_ref __stream)
      : uninitialized_buffer<_Type, device_accessible>(__async_device_memory_resource(__stream), __count)
      , __immutable_buffer_base<device_buffer>()
  {}
};

} // namespace cuda::experimental::async

#endif // __CUDAX_ASYNC_DETAIL_DEVICE_BUFFER
