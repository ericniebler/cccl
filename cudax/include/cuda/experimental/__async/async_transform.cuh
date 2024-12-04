//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__ASYNC_ASYNC_TRANSFORM
#define _CUDAX__ASYNC_ASYNC_TRANSFORM

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/__async/tasks.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{
namespace __async_transform_ns
{
// Launch transform:
//
// The launch transform is a mechanism to transform arguments passed to the
// cudax::launch API prior to actually launching a kernel. This is useful for
// example, to automatically convert contiguous ranges into spans. It is also
// useful for executing per-argument actions before and after the kernel launch.
// A host_vector might want a pre-launch action to copy data from host to device
// and a post-launch action to copy data back from device to host.
//
// The launch transform happens in two steps. First, `cudax::launch` calls
// async_transform on each argument. If the argument has hooked the
// async_transform customization point, this returns a temporary object that
// has the pre-launch action in its constructor and the post-launch action in
// its destructor. The temporaries are all constructed before launching the
// kernel, and they are all destroyed immediately after, at the end of the full
// expression that performs the launch. If the `cudax::launch` argument has not
// hooked the async_transform customization point, then the argument is
// passed through.
//
// The result of async_transform is not necessarily what is passed to the
// kernel though. If async_transform returns an object with a
// `.relocatable_value()` member function, then `cudax::launch` will call that
// function. Its result is what gets passed as an argument to the kernel. If the
// async_transform result does not have a `.relocatable_value()` member
// function, then the async_transform result itself is passed to the kernel.

void cuda_async_transform();

// Types that want to customize `async_transform` should define overloads of
// cuda_async_transform that are find-able by ADL.
struct __fn
{
  template <typename _Arg>
  using __result_t = decltype(cuda_async_transform(::cuda::stream_ref{}, declval<_Arg>()));

  template <typename _Arg>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_HOST_API decltype(auto) operator()(::cuda::stream_ref __stream, _Arg&& __arg) const
  {
    if constexpr (_CUDA_VSTD::_IsValidExpansion<__result_t, _Arg>::value)
    {
      // This call is unqualified to allow ADL
      return cuda_async_transform(__stream, _CUDA_VSTD::forward<_Arg>(__arg));
    }
    else
    {
      return _Arg(_CUDA_VSTD::forward<_Arg>(__arg));
    }
  }
};
} // namespace __async_transform_ns

struct __relocatable_value_fn
{
  template <typename _Arg>
  using __result_t = decltype(declval<_Arg>().relocatable_value());

  template <typename _Arg>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_HOST_API decltype(auto) operator()(_Arg&& __arg) const
  {
    if constexpr (_CUDA_VSTD::_IsValidExpansion<__result_t, _Arg>::value)
    {
      return _CUDA_VSTD::forward<_Arg>(__arg).relocatable_value();
    }
    else
    {
      return _Arg(_CUDA_VSTD::forward<_Arg>(__arg));
    }
  }
};

inline namespace __cpo
{
inline constexpr __async_transform_ns::__fn async_transform{};
inline constexpr __relocatable_value_fn relocatable_value{};
} // namespace __cpo

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2017
#endif // !_CUDAX__ASYNC_ASYNC_TRANSFORM
