//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

#  if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#    include <cuda_runtime_api.h>
#  endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__cuda/ensure_current_device.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

//! @brief cuda_memory_resource uses `cudaMalloc` / `cudaFree` for allocation / deallocation.
//! By default uses device 0 to allocate memory
class cuda_memory_resource
{
private:
  int __device_id_{0};

public:
  //! @brief default constructs a cuda_memory_resource allocating memory on device 0
  cuda_memory_resource() = default;

  //! @brief default constructs a cuda_memory_resource allocating memory on device \p __device_id
  //! @param __device_id The id of the device we are allocating memory on
  constexpr cuda_memory_resource(const int __device_id) noexcept
      : __device_id_(__device_id)
  {}

  //! @brief Allocate device memory of size at least \p __bytes.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throw std::bad_alloc in case of invalid alignment or \c cuda::cuda_error of the returned error code.
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void* allocate(const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    // We need to ensure that we allocate on the right device as `cudaMalloc` always uses the current device
    __ensure_current_device __device_wrapper{__device_id_};

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(::cudaMalloc, "Failed to allocate memory with cudaMalloc.", &__ptr, __bytes);
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  void
  deallocate(void* __ptr, const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment) const noexcept
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFree, "cuda_memory_resource::deallocate failed", __ptr);
    (void) __bytes;
    (void) __alignment;
  }

#    if CUDART_VERSION >= 11020
  //! @brief Asynchronously allocate device memory of size at least \p __bytes
  //! on stream \c __stream.
  //!
  //! The allocation comes from the memory pool associated with the device of
  //! \c __stream.
  //!
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @param __stream The stream establishing the stream ordering contract and
  //!        the memory pool to allocate from.
  //! @throw cuda::cuda_error of the returned error code
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void* allocate_async(const size_t __bytes, const size_t __alignment, stream_ref __stream) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync, "Failed to allocate memory with cudaMallocAsync.", &__ptr, __bytes, __stream.get());
    return __ptr;
  }

  //! @brief Asynchronously deallocate memory pointed to by \p __ptr .
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  //! @param __stream The stream establishing the stream ordering contract.
  void deallocate_async(
    void* __ptr, const size_t __bytes, const size_t __alignment, ::cuda::stream_ref __stream) const noexcept
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeAsync, "cudaFreeAsync failed", __ptr, __stream.get());
    (void) __bytes;
    (void) __alignment;
  }
#    endif // CUDART_VERSION >= 11020

  //! @brief Equality comparison with another \c cuda_memory_resource
  //! @param __other The other \c cuda_memory_resource
  //! @return true, if both resources hold the same device id
  _CCCL_NODISCARD constexpr bool operator==(cuda_memory_resource const& __other) const noexcept
  {
    return __device_id_ == __other.__device_id_;
  }
#    if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c cuda_memory_resource
  //! @param __other The other \c cuda_memory_resource
  //! @return true, if both resources hold different device id's
  _CCCL_NODISCARD constexpr bool operator!=(cuda_memory_resource const& __other) const noexcept
  {
    return __device_id_ != __other.__device_id_;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Equality comparison between a \c cuda_memory_resource and another resource
  //! @param __lhs The \c cuda_memory_resource
  //! @param __rhs The resource to compare to
  //! @return If the underlying types are equality comparable, returns the result of equality comparison of both
  //! resources. Otherwise, returns false.
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(cuda_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_resource&>(__lhs)} == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    if _CCCL_STD_VER <= 2017
  //! @copydoc cuda_memory_resource::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, cuda_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_resource&>(__lhs)} == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  //! @copydoc cuda_memory_resource::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(cuda_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_resource&>(__lhs)} != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  //! @copydoc cuda_memory_resource::operator==<_Resource>(cuda_memory_resource const&, _Resource const&)
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, cuda_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_memory_resource&>(__lhs)} != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Enables the \c device_accessible property
  friend constexpr void get_property(cuda_memory_resource const&, device_accessible) noexcept {}

  //! @brief Checks whether the passed in alignment is valid
  static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_alignment && (default_cuda_malloc_alignment % __alignment == 0);
  }
};
static_assert(resource_with<cuda_memory_resource, device_accessible>, "");
static_assert(async_resource_with<cuda_memory_resource, device_accessible>, "");

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H
