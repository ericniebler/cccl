//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CONTAINER_VECTOR
#define _CUDAX__CONTAINER_VECTOR

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__launch/param_kind.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{
using ::cuda::std::span;
using ::thrust::device_vector;
using ::thrust::host_vector;

template <typename _Ty>
class vector
{
public:
  vector() = default;
  explicit vector(size_t __n)
      : __h_(__n)
  {}

  _Ty& operator[](size_t __i) noexcept
  {
    __dirty_ = true;
    return __h_[__i];
  }

  const _Ty& operator[](size_t __i) const noexcept
  {
    return __h_[__i];
  }

private:
  void sync_host_to_device([[maybe_unused]] ::cuda::stream_ref __str, __detail::__param_kind __p) const
  {
    if (__dirty_)
    {
      if (__p == __detail::__param_kind::_out)
      {
        // There's no need to copy the data from host to device if the data is
        // only going to be written to. We can just allocate the device memory.
        __d_.resize(__h_.size());
      }
      else
      {
        // TODO: use a memcpy async here
        __d_ = __h_;
      }
      __dirty_ = false;
    }
  }

  void sync_device_to_host(::cuda::stream_ref __str, __detail::__param_kind __p) const
  {
    if (__p != __detail::__param_kind::_in)
    {
      // TODO: use a memcpy async here
      __str.sync(); // wait for the kernel to finish executing
      __h_ = __d_;
    }
  }

  template <__detail::__param_kind _Kind>
  class __action //: private __detail::__immovable
  {
    using __cv_vector = ::cuda::std::__maybe_const<_Kind == __detail::__param_kind::_in, vector>;

  public:
    explicit __action(::cuda::stream_ref __str, __cv_vector& __v) noexcept
        : __str_(__str)
        , __v_(__v)
    {
      __v_.sync_host_to_device(__str_, _Kind);
    }

    __action(__action&&) = delete;

    ~__action()
    {
      __v_.sync_device_to_host(__str_, _Kind);
    }

    ::cuda::std::span<_Ty> transformed_argument() const
    {
      return {__v_.__d_.data().get(), __v_.__d_.size()};
    }

  private:
    ::cuda::stream_ref __str_;
    __cv_vector& __v_;
  };

  [[nodiscard]] friend __action<__detail::__param_kind::_inout>
  transform_device_argument(::cuda::stream_ref __str, vector& __v) noexcept
  {
    return __action<__detail::__param_kind::_inout>{__str, __v};
  }

  [[nodiscard]] friend __action<__detail::__param_kind::_in>
  transform_device_argument(::cuda::stream_ref __str, const vector& __v) noexcept
  {
    return __action<__detail::__param_kind::_in>{__str, __v};
  }

  template <__detail::__param_kind _Kind>
  [[nodiscard]] friend __action<_Kind>
  transform_device_argument(::cuda::stream_ref __str, __detail::__box<vector, _Kind> __b) noexcept
  {
    return __action<_Kind>{__str, __b.__val};
  }

  mutable host_vector<_Ty> __h_;
  mutable device_vector<_Ty> __d_{};
  mutable bool __dirty_ = true;
};

} // namespace cuda::experimental

#endif
#endif
