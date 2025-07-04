//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_DEVICE_TRANSFORM
#define _CUDAX__STREAM_DEVICE_TRANSFORM
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace __detail
{
// This function turns rvalues into prvalues and leaves lvalues as is.
template <typename _Tp>
_CCCL_API constexpr auto __ixnay_xvalue(_Tp&& __value) noexcept(__nothrow_movable<_Tp>) -> _Tp
{
  return _CUDA_VSTD::forward<_Tp>(__value);
}
} // namespace __detail

template <typename _Tp>
using __remove_rvalue_reference_t = decltype(__detail::__ixnay_xvalue(_CUDA_VSTD::declval<_Tp>()));

namespace __tfx
{
// Device transform:
//
// The device transform is a mechanism to transform arguments passed to the
// algorithms prior to actually enqueueing work on a stream. This is useful for
// example, to automatically convert contiguous ranges into spans. It is also
// useful for executing per-argument actions before and after the kernel launch.
// A host_vector might want a pre-launch action to copy data from host to device
// and a post-launch action to copy data back from device to host.
//
// The expression `device_transform(stream, arg)` is expression-equivalent to
// the first of the following expressions that is valid:
//
// 1. `cuda_device_transform(stream, arg).relocatable_value()`
// 2. `cuda_device_transform(stream, arg)`
// 3. `arg.relocatable_value()`
// 4. `arg`
void cuda_device_transform();

struct __device_transform_t
{
  // This is used to insert an object into the caller's stack frame.
  template <class _Arg, bool _IsReference = _CUDA_VSTD::is_reference_v<_Arg>>
  struct __storage_for
  {
    constexpr __storage_for() noexcept {}

    ~__storage_for()
    {
      if (__engaged)
      {
        __value.~_Arg();
      }
    }

    template <class... _OtherArgs>
    auto __emplace(_OtherArgs&&... __arg) noexcept(__nothrow_constructible<_Arg, _OtherArgs...>) -> _Arg&&
    {
      _Arg* __ptr = ::new (_CUDA_VSTD::addressof(__value)) _Arg(_CUDA_VSTD::forward<_OtherArgs>(__arg)...);
      __engaged   = true;
      return _CCCL_MOVE(*_CUDA_VSTD::launder(__ptr));
    }

    union
    {
      _Arg __value;
    };
    bool __engaged = false;
  };

  // No need to store reference types since they are not ephemeral.
  template <class _Arg>
  struct __storage_for<_Arg, true>
  {
    auto __emplace(_Arg __arg) noexcept -> _Arg
    {
      return __arg;
    }
  };

  // Types that want to customize `device_transform` should define overloads of
  // cuda_device_transform that are find-able by ADL.
  template <typename _Arg>
  using __transform_result_t = decltype(cuda_device_transform(::cuda::stream_ref{}, _CUDA_VSTD::declval<_Arg>()));

  template <typename _Arg>
  using __relocatable_value_t = decltype(_CUDA_VSTD::declval<_Arg>().relocatable_value());

  // The use of `__storage_for` here is to move the destruction of the object returned from
  // cuda_device_transform into the caller's stack frame. Objects created for default arguments
  // are located in the caller's stack frame. This is so that a use of `device_transform`
  // such as:
  //
  //   kernel<<<grid, block, 0, stream>>>(device_transform(stream, arg));
  //
  // is equivalent to:
  //
  //   kernel<<<grid, block, 0, stream>>>(cuda_device_transform(stream, arg).relocatable_value());
  //
  // where the object returned from `cuda_device_transform` is destroyed after the kernel
  // launch.
  //
  // What I really wanted to do was:
  //
  //   template <typename Arg>
  //   auto operator()(::cuda::stream_ref stream, Arg&& arg, auto&& tmp = cuda_device_transform(arg)) const
  //
  // but sadly that is not valid C++.
  template <typename _Arg>
  [[nodiscard]] auto operator()(::cuda::stream_ref __stream, //
                                _Arg&& __arg,
                                __storage_for<__transform_result_t<_Arg>> __storage = {}) const -> decltype(auto)
  {
    // Calls to cuda_device_transform are intentionally unqualified so as to use ADL.
    if constexpr (__is_instantiable_with_v<__relocatable_value_t, __transform_result_t<_Arg>>)
    {
      return __storage
        .__emplace(__emplace_from{[&] {
          return cuda_device_transform(__stream, _CUDA_VSTD::forward<_Arg>(__arg));
        }})
        .relocatable_value();
    }
    else
    {
      return __storage.__emplace(__emplace_from{[&] {
        return cuda_device_transform(__stream, _CUDA_VSTD::forward<_Arg>(__arg));
      }});
    }
  }

  template <typename _Arg>
  [[nodiscard]] auto operator()(_CUDA_VSTD::__ignore_t, _Arg&& __arg) const -> decltype(auto)
  {
    if constexpr (__is_instantiable_with_v<__relocatable_value_t, _Arg>)
    {
      return _CUDA_VSTD::forward<_Arg>(__arg).relocatable_value();
    }
    else
    {
      return static_cast<_Arg>(_CUDA_VSTD::forward<_Arg>(__arg));
    }
  }
};
} // namespace __tfx

_CCCL_GLOBAL_CONSTANT auto device_transform = __tfx::__device_transform_t{};

template <typename _Arg>
using device_transform_result_t =
  __remove_rvalue_reference_t<_CUDA_VSTD::__call_result_t<__tfx::__device_transform_t, ::cuda::stream_ref, _Arg>>;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // !_CUDAX__STREAM_DEVICE_TRANSFORM
